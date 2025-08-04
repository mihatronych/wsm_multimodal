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

res_setup_translation_model = _b5.setup_translation_model(
    out = True, # Отображение
    runtime = True, # Подсчет времени выполнения
    run = True # Блокировка выполнения
)

# Настройки ядра
_b5.path_to_save_ = './models' # Директория для сохранения файла
_b5.chunk_size_ = 2000000 # Размер загрузки файла из сети за 1 шаг

res_setup_translation_model = _b5.setup_bert_encoder(
    force_reload = False, # Принудительная загрузка файла
    out = True, # Отображение
    runtime = True, # Подсчет времени выполнения
    run = True # Блокировка выполнения
)

res_load_video_model_hc = _b5.load_video_model_hc(
    lang = 'en', # Language selection for models trained on First Impressions V2'en' and models trained on for MuPTA 'ru'
    show_summary = False, # Displaying the formed neural network architecture of the model
    out = True, # Display
    runtime = True, # Runtime count
    run = True # Run blocking
)

# Настройки ядра
_b5.path_to_save_ = './models' # Директория для сохранения файла
_b5.chunk_size_ = 2000000 # Размер загрузки файла из сети за 1 шаг

url = _b5.weights_for_big5_['video']['fi']['hc']['googledisk']

res_load_video_model_weights_hc = _b5.load_video_model_weights_hc(
    url = url, # Полный путь к файлу с весами нейросетевой модели
    force_reload = True, # Принудительная загрузка файла с весами нейросетевой модели из сети
    out = True, # Отображение
    runtime = True, # Подсчет времени выполнения
    run = True # Блокировка выполнения
)

res_load_video_model_deep_fe = _b5.load_video_model_deep_fe(
    show_summary = False, # Displaying the formed neural network architecture of the model
    out = True, # Display
    runtime = True, # Runtime count
    run = True # Run blocking
)

# Настройки ядра
_b5.path_to_save_ = './models' # Директория для сохранения файла
_b5.chunk_size_ = 2000000 # Размер загрузки файла из сети за 1 шаг

url = _b5.weights_for_big5_['video']['fi']['fe']['googledisk']

res_load_video_model_weights_deep_fe = _b5.load_video_model_weights_deep_fe(
    url = url, # Полный путь к файлу с весами нейросетевой модели
    force_reload = True, # Принудительная загрузка файла с весами нейросетевой модели из сети
    out = True, # Отображение
    runtime = True, # Подсчет времени выполнения
    run = True # Блокировка выполнения
)

res_load_video_model_nn = _b5.load_video_model_nn(
    show_summary = False, # Displaying the formed neural network architecture of the model
    out = True, # Display
    runtime = True, # Runtime count
    run = True # Run blocking
)

# Настройки ядра
_b5.path_to_save_ = './models' # Директория для сохранения файла
_b5.chunk_size_ = 2000000 # Размер загрузки файла из сети за 1 шаг

url = _b5.weights_for_big5_['video']['fi']['nn']['googledisk']

res_load_video_model_weights_nn = _b5.load_video_model_weights_nn(
    url = url, # Полный путь к файлу с весами нейросетевой модели
    force_reload = False, # Принудительная загрузка файла с весами нейросетевой модели из сети
    out = True, # Отображение
    runtime = True, # Подсчет времени выполнения
    run = True # Блокировка выполнения
)

res_load_video_models_b5 = _b5.load_video_models_b5(
    show_summary = False, # Displaying the formed neural network architecture of the model
    out = True, # Display
    runtime = True, # Runtime count
    run = True # Run blocking
)

# Настройки ядра
_b5.path_to_save_ = './models' # Директория для сохранения файла
_b5.chunk_size_ = 2000000 # Размер загрузки файла из сети за 1 шаг

url_openness = _b5.weights_for_big5_['video']['fi']['b5']['openness']['googledisk']
url_conscientiousness = _b5.weights_for_big5_['video']['fi']['b5']['conscientiousness']['googledisk']
url_extraversion = _b5.weights_for_big5_['video']['fi']['b5']['extraversion']['googledisk']
url_agreeableness = _b5.weights_for_big5_['video']['fi']['b5']['agreeableness']['googledisk']
url_non_neuroticism = _b5.weights_for_big5_['video']['fi']['b5']['non_neuroticism']['googledisk']

res_load_video_models_weights_b5 = _b5.load_video_models_weights_b5(
    url_openness = url_openness, # Открытость опыту
    url_conscientiousness = url_conscientiousness, # Добросовестность
    url_extraversion = url_extraversion, # Экстраверсия
    url_agreeableness = url_agreeableness, # Доброжелательность
    url_non_neuroticism = url_non_neuroticism, # Нейротизм
    force_reload = False, # Принудительная загрузка файла с весами нейросетевой модели из сети
    out = True, # Отображение
    runtime = True, # Подсчет времени выполнения
    run = True # Блокировка выполнения
)