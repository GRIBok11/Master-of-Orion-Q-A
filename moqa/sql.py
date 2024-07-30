import sqlite3

# Путь к базе данных
database_path = r'd:\hoho\moqa\moqa\vectre_md\chroma.sqlite3'

# Подключение к базе данных
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Определение запроса
# Предполагается, что ключевое поле имеет название 'Header' и данные находятся в столбце 'Header 2'
query = "SELECT DISTINCT string_value FROM embedding_metadata WHERE `key` = ?"

# Значение ключа для фильтрации
key_value = 'Header 2'

# Выполнение запроса
cursor.execute(query, (key_value,))

# Получение всех строк результата
rows = cursor.fetchall()

# Закрытие соединения с базой данных
conn.close()

# Печать результатов
for row in rows:
    print(row)