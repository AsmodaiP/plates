import argparse
import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import keras_ocr


def preprocess_image(image):
    """
    Применяет шумоподавление и выделение границ к изображению.

    :param image: исходное изображение
    :return: обработанное изображение
    """
    # Шумоподавление
    filtered = cv2.bilateralFilter(image, 11, 17, 17)
    # Выделение границ
    edged = cv2.Canny(filtered, 30, 200)
    return edged


def find_contours(image):
    """
    Находит контуры на изображении и сортирует их по убыванию площади.

    :param image: исходное изображение
    :return: список контуров
    """
    contours = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    return contours


def find_document_contour(contours):
    """
    Находит контур документа среди списка контуров.

    :param contours: список контуров
    :return: контур документа или None, если документ не найден
    """
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            return approx
    return None


def recognize_text(image):
    """
    Распознает текст на изображении.

    :param image: исходное изображение
    :return: распознанный текст
    """
    pipeline = keras_ocr.pipeline.Pipeline()
    prediction_groups = pipeline.recognize([image])
    text = []
    for predictions in prediction_groups:
        for prediction in predictions:
            text.append(prediction[0])
    full_text = ' '.join(text)
    return full_text


def main():
    parser = argparse.ArgumentParser(description='Recognize text on an image')
    parser.add_argument('image_path', type=str, help='path to the image')
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: file {args.image_path} not found")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = preprocess_image(gray)
    contours = find_contours(processed)
    document_contour = find_document_contour(contours)
    if document_contour is None:
        print("Error: document contour not found")
        return
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [document_contour], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)
    recognized_text = recognize_text(new_image)
    
    _, axs = plt.subplots(nrows=2, figsize=(20, 20))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[1].imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Document Contour')
    plt.show()
    print(f"Recognized text: {recognized_text}")


if __name__ == '__main__':
    main()
