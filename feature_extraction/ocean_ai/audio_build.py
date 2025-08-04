from oceanai.modules.lab.build import Run

_b5 = Run(
    lang = 'en', # Interface language
    color_simple = '#333', # Plain text color (hexadecimal code)
    color_info = '#1776D2', # The color of the text containing the information (hexadecimal code)
    color_err = '#FF0000', # Error text color (hexadecimal code)
    color_true = '#008001', # Text color containing positive information (hexadecimal code)
    bold_text = True, # Bold text
    num_to_df_display = 30, # Number of rows to display in tables
    text_runtime = 'Runtime', # Runtime text
    metadata = True # Displaying information about library
)

res_load_audio_model_hc = _b5.load_audio_model_hc(
    show_summary = False, # Отображение сформированной нейросетевой архитектуры модели
    out = True, # Отображение
    runtime = True, # Подсчет времени выполнения
    run = True # Блокировка выполнения
)

# Настройки ядра
_b5.path_to_save_ = './models' # Директория для сохранения файла
_b5.chunk_size_ = 2000000 # Размер загрузки файла из сети за 1 шаг

url = _b5.weights_for_big5_['audio']['fi']['hc']['googledisk']

res_load_audio_model_weights_hc = _b5.load_audio_model_weights_hc(
    url = url, # Полный путь к файлу с весами нейросетевой модели
    force_reload = True, # Принудительная загрузка файла с весами нейросетевой модели из сети
    out = True, # Отображение
    runtime = True, # Подсчет времени выполнения
    run = True # Блокировка выполнения
)