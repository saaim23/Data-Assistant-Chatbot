import openai
import asyncio
import streamlit as st
import polars as pl
import os
from dotenv import load_dotenv, find_dotenv
import plotly.express as px
from streamlit_chat import message
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set OpenAI API key
apikey = os.getenv('OPENAI_API_KEY')

# Ensure the API key is set correctly
if not apikey:
    raise ValueError(
        "OpenAI API key not found. Please set it in the .env file.")

# Title
st.title('AI Assistant for Data Science 🤖')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

# Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with an CSV File.*')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> for experiment by Saaim</p>",
               unsafe_allow_html=True)

# Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state


def clicked(button):
    st.session_state.clicked[button] = True


st.button("Let's get started", on_click=clicked, args=[1])


async def main():
    if st.session_state.clicked[1]:
        tab1, tab2 = st.tabs(["Data Analysis and Data Science", "ChatBox"])

        with tab1:
            user_csv = st.file_uploader("Upload your file here", type="csv")
            if user_csv is not None:
                user_csv.seek(0)
                df = pl.read_csv(user_csv, null_values=['null'])
                df = df.to_pandas()

                # llm model
                llm = OpenAI(api_key=apikey)

                # Function sidebar
                @st.cache_resource
                async def fetch_steps_eda():
                    response = await openai.ChatCompletion.acreate(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "What are the steps of EDA?"}
                        ]
                    )
                    steps_eda = {"key": response.choices[0].message['content']}
                    return steps_eda

                @st.cache_data
                async def data_science_framing():
                    response = await openai.ChatCompletion.acreate(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Write a couple of paragraphs about the importance of framing a data science problem appropriately."}
                        ]
                    )
                    return response.choices[0].message['content']

                @st.cache_data
                async def algorithm_selection():
                    response = await openai.ChatCompletion.acreate(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Write a couple of paragraphs about the importance of considering more than one algorithm when trying to solve a data science problem."}
                        ]
                    )
                    return response.choices[0].message['content']

                # Pandas agent
                pandas_agent = create_pandas_dataframe_agent(
                    llm, df, verbose=True, allow_dangerous_code=True)

                # Functions main
                @st.cache_data
                async def function_agent():
                    st.write("**Data Overview**")
                    st.write("The first rows of your dataset look like this:")
                    st.write(df.head())
                    st.write("**Data Cleaning**")
                    columns_df = await pandas_agent.run("What are the meanings of the columns?")
                    st.write(columns_df)
                    missing_values = await pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
                    st.write(missing_values)
                    duplicates = await pandas_agent.run("Are there any duplicate values and if so where?")
                    st.write(duplicates)
                    st.write("**Data Summarisation**")
                    st.write(df.describe())
                    correlation_analysis = await pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
                    st.write(correlation_analysis)
                    outliers = await pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                    st.write(outliers)
                    new_features = await pandas_agent.run("What new features would be interesting to create?")
                    st.write(new_features)
                    return

                @st.cache_data
                async def function_question_variable(df, user_question_variable, pandas_agent):
                    # Plot the data
                    st.plotly_chart(px.line(df, y=[user_question_variable]))

                    # Get summary statistics
                    summary_statistics = await pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
                    st.write(summary_statistics)

                    # Check for normality or specific distribution shapes
                    normality = await pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
                    st.write(normality)

                    # Assess the presence of outliers
                    outliers = await pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
                    st.write(outliers)

                    # Analyse trends, seasonality, and cyclic patterns
                    trends = await pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
                    st.write(trends)

                    # Determine the extent of missing values
                    missing_values = await pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
                    st.write(missing_values)

                    return

                # Main
                st.header('Exploratory data analysis')
                st.subheader('General information about the dataset')

                with st.sidebar:
                    with st.expander('What are the steps of EDA'):
                        st.write(await fetch_steps_eda())

                await function_agent()

                st.subheader('Variable of study')
                user_question_variable = st.text_input(
                    'What variable are you interested in')
                if user_question_variable is not None and user_question_variable != "":
                    await function_question_variable(df, user_question_variable, pandas_agent)

                    st.subheader('Further study')

                if user_question_variable:
                    user_question_dataframe = st.text_input(
                        "Is there anything else you would like to know about your dataframe?")
                    if user_question_dataframe is not None and user_question_dataframe not in ("", "no", "No"):
                        await function_question_variable(df, user_question_dataframe, pandas_agent)
                    if user_question_dataframe in ("no", "No"):
                        st.write("")

                    if user_question_dataframe:
                        st.divider()
                        st.header("Data Science Problem")
                        st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's important that we reframe our business problem into a data science problem.")

                        with st.sidebar:
                            with st.expander("The importance of framing a data science problem appropriately"):
                                st.caption(await data_science_framing())

                        prompt = st.text_area(
                            'What is the business problem you would like to solve?')

                        if prompt:
                            wiki_research = await wiki(prompt)
                            my_data_problem = (await chains_output(prompt, wiki_research))[0]
                            my_model_selection = (await chains_output(prompt, wiki_research))[1]
                            with st.sidebar:
                                with st.expander("Is one algorithm enough?"):
                                    st.caption(await algorithm_selection())

                            st.write(my_data_problem)
                            st.write(my_model_selection)

                            formatted_list = list_to_selectbox(
                                my_model_selection)
                            selected_algorithm = st.selectbox(
                                "Select Machine Learning Algorithm", formatted_list)

                            if selected_algorithm is not None and selected_algorithm != "Select Algorithm":
                                st.subheader("Solution")
                                solution = await python_solution(my_data_problem, selected_algorithm, user_csv)
                                st.write(solution)

        with tab2:
            st.header("ChatBox")
            st.write("🤖 Welcome to the AI Assistant ChatBox!")
            st.write("Got burning questions about your data science problem or need help navigating the intricacies of your project? You're in the right place! Our chatbot is geared up to assist you with insights, advice, and solutions. Just type in your queries, and let's unravel the mysteries of your data together! 🔍💻")

            st.write("")

            if 'responses' not in st.session_state:
                st.session_state['responses'] = ["How can I assist you?"]
            if 'requests' not in st.session_state:
                st.session_state['requests'] = []

            llm = ChatOpenAI(model_name="gpt-4", openai_api_key=apikey)

            if 'buffer_memory' not in st.session_state:
                st.session_state.buffer_memory = ConversationBufferWindowMemory(
                    k=3, return_messages=True)

            system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
            and if the answer is not contained within the text below, say 'I don't know'""")
            human_msg_template = HumanMessagePromptTemplate.from_template(
                template="{input}")
            prompt_template = ChatPromptTemplate.from_messages(
                [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

            conversation = ConversationChain(
                memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

            response_container = st.container()
            textcontainer = st.container()

            with textcontainer:
                query = st.text_input(
                    "Hello! How can I help you?", key="input")
                if query:
                    with st.spinner("thinking..."):
                        conversation_string = get_conversation_string()
                        refined_query = query_refiner(
                            conversation_string, query)
                        st.subheader("Refined Query:")
                        st.write(refined_query)
                        context = find_match(refined_query)
                        response = await conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                    st.session_state.requests.append(query)
                    st.session_state.responses.append(response)

            with response_container:
                if st.session_state['responses']:
                    for i in range(len(st.session_state['responses'])):
                        message(st.session_state['responses'][i], key=str(i))
                        if i < len(st.session_state['requests']):
                            message(
                                st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

# Run the main function
asyncio.run(main())
