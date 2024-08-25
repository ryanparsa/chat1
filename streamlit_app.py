import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Optionally, specify your own session_state key for storing messages
messages_history = StreamlitChatMessageHistory()

default_system_prompt_filename = 'ChatbotSystemPromot1_Llama3.md'


def get_default_system_prompt(filename: str = default_system_prompt_filename) -> str:
    return open(filename, 'r').read()


st.title('AI Confidence Coach')
st.write('This is a web application that uses AI to help you build confidence.')


def reset_history():
    messages_history.clear()
    messages_history.add_ai_message('How can I help you?')


with st.sidebar:
    llm = st.selectbox('Select a LLM', ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini', 'groq', 'gemini'],
                       on_change=reset_history)
    llm_token = st.text_input('Token', help='Please provide the token', on_change=reset_history)
    llm_prompt = st.text_area(
        label='Prompt',
        value=get_default_system_prompt(),
        placeholder=f'this prompt helps the model to generate the text based on what you want to achieve',
        help=f'Please provide a prompt to help the model generate the text',
        height=400,
        on_change=reset_history)

# If the user has filled in the required fields, we can proceed
if llm_token and llm_prompt and llm:

    if llm in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']:
        llm_model = ChatOpenAI(model=llm, api_key=llm_token)
    elif llm == 'groq':
        llm_model = ChatGroq(model="llama3-8b-8192", api_key=llm_token)
    elif llm == 'gemini':
        llm_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=llm_token)

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', llm_prompt),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{question}'),
        ]
    )

    chain = prompt | llm_model

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: messages_history,
        input_messages_key='question',
        history_messages_key='history',
    )

    if len(messages_history.messages) == 0:
        reset_history()

    for msg in messages_history.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input():
        st.chat_message('human').write(prompt)

        # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
        config = {'configurable': {'session_id': 'any'}}
        response = chain_with_history.invoke({'question': prompt}, config)
        st.chat_message('ai').write(response.content)
else:
    st.write('Please fill in the required fields and click submit.')
