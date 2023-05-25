import cv2
import pytesseract

def encontrarRoiPlaca(source):
    img = cv2.imread(source)
    cv2.imshow("img", img)
    
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("cinza", img)

    _, bin = cv2.threshold(cinza, 100, 255, cv2.THRESH_BINARY)
   # cv2.imshow("binary", img)
    desfoque = cv2.GaussianBlur(bin, (3, 3), 0)
    cv2.imshow("defoque", desfoque)
    
    
    laplacian = cv2.Laplacian(desfoque,cv2.CV_64F)
    sobelx = cv2.Sobel(desfoque,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(desfoque,cv2.CV_64F,0,1,ksize=5)
    cv2.imshow('laplacian', laplacian)
    
    
    t_lower = 100 # Lower Threshold
    t_upper = 100 # Upper threshold
    aperture_size = 3 # Aperture size
    L2Gradient = True # Boolean
    
    # Applying the Canny Edge filter 
    # with Aperture Size and L2Gradient
    edge = cv2.Canny(desfoque, t_lower, t_upper,
                    apertureSize= aperture_size,
                    L2gradient = L2Gradient ) 
    
    cv2.imshow('original', img)
    cv2.imshow('edge', edge)
  


    contornos, hierarquia = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contornos, -1, (0, 255, 0), 1)

    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        if perimetro > 120:
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(aprox) == 4:
                (x, y, alt, lar) = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + alt, y + lar), (0, 255, 0), 2)
                roi = img[y:y + lar, x:x + alt]
                cv2.imwrite('output/roi.png', roi)

    cv2.imshow("contornos", img)


def preProcessamentoRoiPlaca():
    img_roi = cv2.imread("output/roi.png")

    if img_roi is None:
        return

    resize_img_roi = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Converte para escala de cinza
    img_cinza = cv2.cvtColor(resize_img_roi, cv2.COLOR_BGR2GRAY)

    # Binariza imagem
    _, img_binary = cv2.threshold(img_cinza, 70, 255, cv2.THRESH_BINARY)

    # Desfoque na Imagem
    img_desfoque = cv2.GaussianBlur(img_binary, (5, 5), 0)

    # Grava o pre-processamento para o OCR
    cv2.imwrite("output/roi-ocr.png", img_desfoque)

    #cv2.imshow("ROI", img_desfoque)

    cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return img_desfoque


def ocrImageRoiPlaca():
    image = cv2.imread("output/roi-ocr.png")

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

    saida = pytesseract.image_to_string(image, lang='eng', config=config)#aparente mente é aqui

    return saida


if __name__ == "__main__":
    encontrarRoiPlaca("resource/carro3.jpg")

    pre = preProcessamentoRoiPlaca()

    # ocr = ocrImageRoiPlaca()#estadando erro

    # print(ocr)
