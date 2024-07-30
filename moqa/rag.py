def determine_topic(question):
    # Ваша функция для определения темы
    # Возвращает строку, представляющую тему вопроса
    pass

def create_chain(retriever, determine_topic_func):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Q&A helper for the documentation, trained to answer questions from the Master of Orion manual."
                "\n\nThe relevant documents will be retrieved in the following messages.",
            ),
            ("system", "{context}"),
            ("human", "{question}"),
        ]
    )

    response_generator = prompt | model | StrOutputParser()
    chain = (
        {
            "context": lambda inputs: retriever.retrieve(determine_topic_func(inputs["question"])),
            "question": itemgetter("question"),
        }
        | response_generator
    )
    return chain

def main():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Загрузка и разбиение данных на документы
    loader = TextLoader("moo2.md", encoding="utf-8")
    data = loader.load()
    data_str = "\n".join([doc.page_content for doc in data])
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_o)
    md_header_splits = markdown_splitter.split_text(data_str)

    # Создание хранилища и извлекатель
    vectorstore = Chroma.from_documents(documents=md_header_splits, embedding=embedding_function, persist_directory="vectre_md")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Создание цепочки с использованием функции определения темы
    chain = create_chain(retriever, determine_topic)

    # Дальнейшая логика, например, выполнение запросов с использованием chain

if __name__ == "__main__":
    main()
