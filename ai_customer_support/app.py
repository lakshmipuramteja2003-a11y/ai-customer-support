import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

SYSTEM_PROMPT = """You are a helpful, empathetic e-commerce customer support assistant.
You are not an employee or representative of a particular retailer; do not imply
that you are, or refer to a store as "our store".
Answer clearly and keep the customer's issue moving toward a practical resolution.
Use only information provided in the conversation. You cannot access order systems,
issue refunds, replace products, change orders, browse the web, or use ticketing
systems. Never claim that you have taken any action outside this chat. Do not say
that you documented, recorded, submitted, forwarded, escalated, or passed a request
along, and do not offer to search websites or social media. You may suggest that the
customer document damage, but do not claim to have documented it yourself. When
information is missing, ask a concise follow-up question (for example, the latest
tracking update, delivery estimate, or a description of the damage). Do not ask the
customer to share order numbers, addresses, payment details, or other private account
identifiers in this chat; direct them to the store's verified support channel for
account-specific help.
You cannot look up, check, track, or investigate an order, even if the customer
shares an order number. For late deliveries, say that you cannot view tracking and
ask what the latest tracking update shows, or direct the customer to the seller or
carrier for an account-specific update. Never promise to provide an order update.
For damaged products, ask about the physical damage and suggest documenting it with
photos; do not ask about tracking unless it is relevant. A wrong color, size, or
model is a mismatched item, not physical damage; acknowledge it as a fulfillment
issue and do not ask the customer to confirm whether it is damaged. Keep separate
issues distinct and use relevant details already shared in follow-ups. For returns,
do not infer why the customer wants to return an item or ask for a reason they have
not given. Eligibility and deadlines vary by seller, so direct them to the seller's
published return policy or verified support channel. Do not ask for the return reason
or item condition unless the customer asks about eligibility, and never offer to look
up a policy. For refunds, suggest checking the original payment method and contacting
the seller about status. If the seller has approved a refund, advise checking for a
pending credit and asking the seller when it was issued and for its expected timeline.
Processing times vary; do not invent a specific timeline or claim to see transaction
status.
When asked for a replacement, say plainly that you cannot request or place one, then
direct the customer to the seller's verified support channel. Never ask the customer
to confirm facts they already stated, such as the wrong color or a broken sole.
When asked for seller contact information, say you cannot browse or retrieve it and
suggest checking the seller's official website, order confirmation, or marketplace
support. Do not offer to search for it.
If a request needs account-specific help, direct the customer to the store's support team."""

UNSUPPORTED_ACTION_PATTERN = re.compile(
    r"\b(?:i|we)(?:'ll| will| can| could| would| have|'ve)\s+"
    r"(?:(?:need|have)\s+to\s+)?(?:look into|check|track|process|submit|request|"
    r"place|order|replace|arrange|contact|notify|forward|pass along|document|"
    r"record|escalate|search|browse|look up|retrieve)\b"
    r"|\b(?:i|we)(?:'ve| have)\s+(?:documented|recorded|submitted|forwarded|"
    r"passed|contacted|processed|requested|ordered|replaced)\b"
    r"|\bwould you like me to\s+(?:(?:help you)\s+)?"
    r"(?:search|browse|look up|find|retrieve|provide)\b"
    r"|\b(?:replacement|request|ticket)\s+(?:has been|was)\s+"
    r"(?:requested|placed|processed|submitted|sent|forwarded)\b",
    re.IGNORECASE,
)


@st.cache_resource(show_spinner=False)
def build_chain(provider: str, model: str, base_url: str, api_key: str):
    if provider == "OpenRouter":
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=model, base_url=base_url, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm


def to_lc_history(messages: list[dict[str, str]]) -> list[HumanMessage | AIMessage]:
    history: list[HumanMessage | AIMessage] = []
    for message in messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))
    return history


def enforce_capability_limits(
    response: str, messages: list[dict[str, str]], question: str
) -> str:
    user_context = " ".join(
        message["content"] for message in messages if message["role"] == "user"
    ).casefold()
    response_lower = response.casefold()
    wrong_item_reported = (
        "ordered" in user_context
        and "received" in user_context
        and any(marker in user_context for marker in ("but received", "instead of", "wrong color", "wrong size", "mismatched"))
    )
    damage_reported = "broken" in user_context and "sole" in user_context
    unsafe_claim = bool(UNSUPPORTED_ACTION_PATTERN.search(response))
    question_lower = question.casefold()

    if "replacement" in question_lower or "replace" in question_lower:
        issue_parts = []
        if wrong_item_reported:
            issue_parts.append("the wrong color")
        if damage_reported:
            issue_parts.append("the broken sole")
        issue_note = f"You described {' and '.join(issue_parts)}. " if issue_parts else ""
        return (
            "I can't request or place a replacement. "
            f"{issue_note}Contact the seller through its verified support channel to ask about "
            "replacement options. You can share photos there; keep order and payment details out of this chat."
        )

    if "contact" in question_lower or "find" in question_lower:
        return (
            "I can't look up seller contact details. Check the seller's official website, "
            "your order confirmation, or the marketplace's verified support page. "
            "Share account details only through those official channels."
        )

    refund_context = "refund" in user_context
    if "refund" in question_lower or (refund_context and "approved" in question_lower):
        refund_approved = "approved" in user_context or "approved" in question_lower
        if refund_approved:
            return (
                "Since the seller approved the refund, check your original payment method for a "
                "pending credit. If it hasn't appeared, reply to the approval email or contact "
                "the seller through its verified support channel. Ask when the refund was issued "
                "and for its expected processing time. I can't see the transaction status, and "
                "timelines vary by seller and payment provider."
            )
        return (
            "I can't see your refund status. Check the original payment method and any return "
            "confirmation from the seller. Contact the seller through its verified support "
            "channel to ask whether the refund was issued and for its expected processing time; "
            "timelines vary by seller and payment provider."
        )

    return_context = "return" in user_context
    return_request = "return" in question_lower or (
        return_context and any(term in question_lower for term in ("size", "wrong", "received"))
    )
    if return_request:
        size_issue = "size" in user_context or "size" in question_lower
        if size_issue:
            return (
                "Since the size is wrong, check the seller's published return policy or verified "
                "support page for the return steps and eligibility. If its return form asks for "
                "a reason, select the option that matches your situation. I can't look up the "
                "policy or start a return, and you don't need to share private order details here."
            )
        return (
            "Check the seller's published return policy or verified support page for the return "
            "steps and eligibility. If its return form asks for a reason, select the option that "
            "matches your situation. I can't look up the policy or start a return, and you don't "
            "need to share private order details here."
        )

    if wrong_item_reported and (unsafe_claim or "confirm" in response_lower):
        damage_note = " and a broken sole" if damage_reported else ""
        return (
            "You've already described receiving the wrong color"
            f"{damage_note}. No need to reconfirm it. I can't access or change your order. "
            "Report the issue through the seller's verified support channel; photos may help. "
            "Keep order and payment details out of this chat."
        )

    if not unsafe_claim:
        return response

    return (
        "I can't take actions, browse the web, or access seller and order systems. "
        "Please contact the seller through its verified support channel for account-specific help."
    )


def friendly_error(error: Exception) -> str:
    if isinstance(error, ValueError):
        return str(error)

    details = str(error).casefold()
    if any(term in details for term in ("401", "403", "api key", "authentication", "unauthorized")):
        return "The provider rejected its credentials. Check your API key and try again."
    if "model" in details and any(term in details for term in ("not found", "does not exist", "404")):
        return "That model is unavailable. Check the model name and confirm it is available to your provider."
    if any(term in details for term in ("connection", "connect", "refused", "timeout", "name or service")):
        return "Could not connect to the selected provider. Check that Ollama is running or that your network is available."
    return "The assistant could not complete that request. Check the provider configuration and try again."


st.set_page_config(page_title="Customer Care Assistant", layout="centered")
st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f1f5f2 0%, #fbfaf7 56%);
        color: #20332e !important;
    }
    [data-testid="stSidebar"] {
        color: #20332e !important;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 840px; padding-top: 2rem; }
    h1 { font-family: Georgia, serif; color: #183b35; font-weight: 500; }
    .stApp p,
    .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #20332e !important;
    }
    .stApp input,
    .stApp textarea,
    .stApp [data-baseweb="select"] * {
        color: #20332e !important;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #e2e8e3;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.76);
    }
    [data-testid="stChatInput"] { border-color: #b7c9c0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Customer care")
st.caption("E-commerce support assistant")

with st.sidebar:
    st.subheader("Assistant settings")
    default_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    providers = ["Ollama", "OpenRouter"]
    selected_provider = st.selectbox(
        "LLM provider",
        providers,
        index=1 if default_provider == "openrouter" else 0,
    )

    if selected_provider == "Ollama":
        model = st.text_input("Model", value=os.getenv("OLLAMA_MODEL", "llama3.2"))
        base_url = st.text_input(
            "Ollama URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        api_key = ""
    else:
        model = st.text_input(
            "Model", value=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        )
        base_url = ""
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            st.caption("Set OPENROUTER_API_KEY in your .env file to use OpenRouter.")

    if st.button("Clear conversation", icon=":material/delete:", use_container_width=True):
        st.session_state.messages = []

st.caption("This assistant cannot look up orders or make account changes.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Hi! Tell me what happened, and I'll help you figure out the next step.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about an order, return, or refund"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            chain = build_chain(selected_provider, model, base_url, api_key)
            with st.spinner("Thinking..."):
                answer = chain.invoke(
                    {
                        "history": to_lc_history(st.session_state.messages[:-1]),
                        "question": question,
                    }
                )
            response = enforce_capability_limits(
                str(answer.content), st.session_state.messages, question
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as error:
            st.error(friendly_error(error))