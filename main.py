import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO


def infer_image(model, image, conf):
    '''
    Производит инференс модели
    Parameters
    ----------
    model : yolo
        Инференс.
    image : numpy.ndarray
        Изображение для предикта.
    conf : float
        Минимальная уверенность в маске.

    Returns
    -------
    ultralytics.engine.results.Results
        Результат, полученный с помощью модели.
    '''
    # Инференс
    return model(image, conf=conf)[0]


def create_mask(image, results):
    '''
    Переводит предсказание модели в бинарную маску
    Parameters
    ----------
    image : numpy.ndarray
        Изображение для предикта.
    results : ultralytics.engine.results.Results
        Результат, полученный с помощью модели.

    Returns
    -------
    mask : numpy.ndarray
        Маска для изображения.
    '''
    height, width = image.shape[:2]

    # Создаем пустую маску с черным фоном
    mask = np.zeros((height, width), dtype=np.uint8)

    # Проходим по результатам и создаем маску
    masks = results.masks  # Получаем маски из результатов
    if masks is not None:
        for mask_array in masks.data:  # Получаем маски как массивы
            mask_i = mask_array.numpy()  # Преобразуем маску в numpy массив

            # Изменяем размер маски под размер оригинального изображения
            mask_i_resized = cv2.resize(mask_i, (width, height), interpolation=cv2.INTER_LINEAR)

            # Накладываем маску на пустую маску (255 для белого)
            mask[mask_i_resized > 0] = 255

    return mask


def degree_pollution(image, results, mode='max'):
    '''
    Считает степень загрязнения с помощью дисперсии лапласиана.
    Большее значение соответсвует меньшей степени загрязнения.
    Parameters
    ----------
    image : numpy.ndarray
        Изображение для предикта.
    results : ultralytics.engine.results.Results
        Результат, полученный с помощью модели.
    mode : str
        Режим получения одной степени загрязнения изображения по всем bbox. По умолчанию 'max'.

    Returns
    -------
    degree : float | None
        Степень загрязнения
    '''
    # Получаем координаты bbox
    xy = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    arr_degree = np.array([])
    # Применяем лапласиан к bbox
    for n in range(xy.shape[0]):
        cropped_image = image[xy[n][1]:xy[n][3],xy[n][0]:xy[n][2]]
        area = cropped_image.shape[0]*cropped_image.shape[1]
        if area > 0:
            arr_degree = np.append(arr_degree, cv2.Laplacian(cropped_image, cv2.CV_64F).var()/area)
    # Использование режима
    if len(arr_degree):
        if mode == 'max':
            degree = np.max(arr_degree)
        elif mode == 'mean':
            degree = np.mean(arr_degree)
        return degree
    else:
        return None


def process_image(image, model_path, calculate_degree=True):
    '''
    Производит анализ изображения, строит карту загрязнения и <делает некоторые другие фичи>
    Parameters
    ----------
    image : numpy.ndarray
        Изображение для предикта.
    model_path : str
        .
    calculate_degree : bool
        Нужно ли считать степень загрязнения.

    Returns
    -------
    mask : numpy.ndarray
        Маска для изображения.
    '''
    height, width = image.shape[:2]
    is_clear = False
    # Применяется метод дисперсии лапласиана
    if height > 600: #для больших картинок
        if cv2.Laplacian(image, cv2.CV_64F).var() >= 4000:
            is_clear = True
    elif cv2.Laplacian(image, cv2.CV_64F).var() >= 200: #для 800х600
        is_clear = True
    # Применяется нейросетевая модель
    model = YOLO(model_path)
    if is_clear:
        results = infer_image(model, image, 0.9) # скорее всего изображение чистое
        mask = create_mask(image, results)
    else:
        results = infer_image(model, image, 0.1) # в изображении не уверены
        mask = create_mask(image, results)
    # Производится постобработка масок
    kernel_7 = np.ones((7,7))
    kernel_3 = np.ones((3,3))
    mask = cv2.erode(cv2.dilate(mask, kernel_7), kernel_3)
    if calculate_degree:
        return mask, degree_pollution(image, results)
    return mask