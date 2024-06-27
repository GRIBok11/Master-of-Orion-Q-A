from dotenv import load_dotenv
import os

# Загрузка переменных из .env файла
load_dotenv()

# Получение значений переменных окружения
database_url = os.getenv('DATABASE_URL')
secret_key = os.getenv('SECRET_KEY')
debug_mode = os.getenv('DEBUG', 'False')  # По умолчанию 'False'

# Преобразование переменной DEBUG к логическому типу
debug_mode = debug_mode.lower() in ['true', '1', 't', 'yes', 'y']

print(f'Database URL: {database_url}')
print(f'Secret Key: {secret_key}')
print(f'Debug Mode: {debug_mode}')
