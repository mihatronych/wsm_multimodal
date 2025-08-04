from .video_build import _b5

from glob import glob
sets = glob('datasets/*/*labels/')
sets[:1]

emo_dict = {0: "neutral", 
            1: "happiness", 
            2: "sadness", 
            3: "surprise", 
            4: "fear", 
            5: "disgust", 
            6: "anger"}

for s in sets:
    # Настройки ядра
    _b5.path_to_dataset_ = s # Директория набора данных
    # Директории не входящие в выборку
    _b5.ignore_dirs_ = []
    # Названия ключей для DataFrame набора данных
    _b5.keys_dataset_ = ['Path', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Non-Neuroticism']
    _b5.ext_ = ['.mp4'] # Расширения искомых файлов
    _b5.path_to_logs_ = f'{_b5.path_to_dataset_}/logs_seg' # Директория для сохранения LOG файлов

    # Полный путь к файлу с верными предсказаниями для подсчета точности
    url_accuracy = _b5.true_traits_['fi']['googledisk']

    res_get_video_union_predictions = _b5.get_video_union_predictions(
        depth = 2,         # Глубина иерархии для получения аудио и видеоданных
        recursive = False, # Рекурсивный поиск данных
        reduction_fps = 5, # Понижение кадровой частоты
        window = 10,       # Размер окна сегмента сигнала (в секундах)
        step = 5,          # Шаг сдвига окна сегмента сигнала (в секундах)
        lang = 'en',
        accuracy = False,   # Вычисление точности
        url_accuracy = url_accuracy,
        logs = True,       # При необходимости формировать LOG файл
        out = True,        # Отображение
        runtime = True,    # Подсчет времени выполнения
        run = True         # Блокировка выполнения
    )