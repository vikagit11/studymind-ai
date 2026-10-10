
evaluation_questions = [
    {
        "question": "Почему возникает переобучение?",
        "relevant_pages": {44},
    },
    {
        "question": "Что такое классификация и регрессия?",
        "relevant_pages": {40},
    },
    {
        "question": "Что такое категориальные переменные?",
        "relevant_pages": {229},
    },
]

for item in evaluation_questions:
    print("Вопрос:", item["question"])
    print("Релевантные страницы:", item["relevant_pages"])
    print()
