from app.query_transformer import QueryTransformer


def main():

    transformer = QueryTransformer()


    queries = [
        "What is the company's work-from-home allowance?",
        "How many vacation days do I get?",
        "Does the company provide dental insurance?"
    ]


    for query in queries:

        transformed = transformer.transform(
            query
        )


        print("\nOriginal:")
        print(query)

        print("\nTransformed:")
        print(transformed)

        print("\n" + "-" * 60)


if __name__ == "__main__":

    main()