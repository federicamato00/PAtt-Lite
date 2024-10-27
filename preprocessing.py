import os
import cv2
import h5py
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import mediapipe as mp
import copy

def visualize_intermediate_steps(original, gray, equalized, normalized,background_removed, processed,save_path):
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Gray Image')
    axes[1].axis('off')
    
    axes[2].imshow(equalized, cmap='gray')
    axes[2].set_title('CLAHE')
    axes[2].axis('off')

    # axes[3].imshow(blurred, cmap='gray')
    # axes[3].set_title('Gaussian Blur')
    # axes[3].axis('off')

    axes[3].imshow(normalized, cmap='gray')
    axes[3].set_title('Normalized')
    axes[3].axis('off')

    axes[4].imshow(background_removed, cmap='gray')
    axes[4].set_title('Background Removed')
    axes[4].axis('off')

    axes[5].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[5].set_title('Processed Image')
    axes[5].axis('off')
    title_new = dataset_name + '_numClasses7_preprocessing.png'
    plot_save_path = os.path.join(save_path, title_new)
    plt.savefig(plot_save_path)
    plt.show()


def apply_unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def convert_to_gray(image):
    copy_image = copy.deepcopy(image)
    return cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)

def apply_clahe(gray_image, clip_limit=0.01):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    copy_image = copy.deepcopy(gray_image)
    return clahe.apply(copy_image)

def apply_gaussian_blur(equalized_image, sigma=1.0):
    copy_image = copy.deepcopy(equalized_image)
    return cv2.GaussianBlur(copy_image, (5, 5), sigma)

def normalize_image(blurred_image):
    copy_image = copy.deepcopy(blurred_image)
    return cv2.normalize(copy_image, None, 0, 255, cv2.NORM_MINMAX)

def convert_to_bgr(thresholded_image):
    # Controlla il numero di canali dell'immagine
    copy_image = copy.deepcopy(thresholded_image)
    if len(thresholded_image.shape) == 2:
        return cv2.cvtColor(copy_image, cv2.COLOR_GRAY2BGR)
    elif len(thresholded_image.shape) == 3 and thresholded_image.shape[2] == 1:
        return cv2.cvtColor(copy_image, cv2.COLOR_GRAY2BGR)
    else:
        return copy_image


def remove_background(normalized):
    image_new = copy.deepcopy(normalized)
    # Controlla il numero di canali dell'immagine
    if len(image_new.shape) == 2:
        # Converte l'immagine in scala di grigi in RGB
        image_new = cv2.cvtColor(image_new, cv2.COLOR_GRAY2RGB)
    
    # Inizializza MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Converti l'immagine in RGB (se non lo è già)
    if image_new.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_new

    # Rileva il volto
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # Ottieni le coordinate del riquadro del volto
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image_new.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Ritaglia il volto dall'immagine
                face_only = image_new[y:y+h, x:x+w]
                if face_only.size == 0:
                    return normalized
                
                # Ridimensiona il volto ritagliato alle dimensioni originali dell'immagine
                face_resized = cv2.resize(face_only, (iw, ih), interpolation=cv2.INTER_LINEAR)

                return face_resized

    # Se non viene rilevato alcun volto, restituisci l'immagine originale
    return normalized


def preprocess_image(image, clip_limit=0.01, sigma=1.0, return_intermediate=False):
    image_copy = copy.deepcopy(image)  # Assicurati di fare una copia dell'immagine originale
    if return_intermediate:
        plt.imshow(image_copy)
        plt.savefig('original.png')
    gray = convert_to_gray(image_copy)
    if return_intermediate:
        plt.imshow(gray, cmap='gray')
        plt.savefig('gray.png')
    equalized = apply_clahe(copy.deepcopy(gray) , clip_limit)
    if return_intermediate:
        plt.imshow(equalized, cmap='gray')
        plt.savefig('equalized.png')
    # blurred = apply_gaussian_blur(equalized.copy(), sigma)
    normalized = normalize_image(copy.deepcopy(equalized) )
    if return_intermediate:
        plt.imshow(normalized, cmap='gray')
        plt.savefig('normalized.png')
    background_removed = remove_background(copy.deepcopy(normalized) )
    if return_intermediate:
        plt.imshow(background_removed, cmap='gray')
        plt.savefig('background_removed.png')
    processed = convert_to_bgr(copy.deepcopy(background_removed) )
    if return_intermediate:
        plt.imshow(processed)
        plt.savefig('processed.png')
    
    return processed

def load_and_preprocess_images(images, clip_limit=0.01, sigma=1.0):
    preprocessed_images = []
    count = 0
    for image in images:
        image_copy = copy.deepcopy(image)
        if count == 0:
             return_intermediate = True
             count +=1
        else:
            return_intermediate = False
        preprocessed_image = preprocess_image(image_copy, clip_limit, sigma, return_intermediate)
        preprocessed_images.append(preprocessed_image)
    return np.array(preprocessed_images)

def load_data_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        X_train = np.array(f['X_train'])
        y_train = np.array(f['y_train'])
        X_val = np.array(f['X_val'])
        y_val = np.array(f['y_val'])
        X_test = np.array(f['X_test'])
        y_test = np.array(f['y_test'])
    return {'train': X_train, 'val': X_val, 'test': X_test}, {'train': y_train, 'val': y_val, 'test': y_test}

def save_data_to_h5(file_path, data, labels):
    with h5py.File(file_path, 'w') as f:
        for split in data.keys():
            f.create_dataset(f'X_{split}', data=data[split])
            f.create_dataset(f'y_{split}', data=labels[split])

# Carica il dataset

dataset_name = 'CK+'
file_name = dataset_name.lower() + '.h5'
dataset_path = os.path.join('datasets',dataset_name, dataset_name+'_numClasses7',file_name )  # Sostituisci con il percorso del tuo file .h5
data, labels = load_data_from_h5(dataset_path)

# Applica il preprocessing alle immagini
data['train'] = load_and_preprocess_images(data['train'])
data['val'] = load_and_preprocess_images(data['val'])
data['test'] = load_and_preprocess_images(data['test'])

# Salva le immagini preprocessate in un nuovo file .h5
processed_file = dataset_name.lower() + '_preprocessed.h5'
save_path = os.path.join('datasets', 'processed', dataset_name+'_numClasses7',processed_file )  # Sostituisci con il percorso del nuovo file .h5
save_data_to_h5(processed_file, data, labels)

print(f"Preprocessed dataset salvato in {processed_file}")

save_path = os.path.join('classify',dataset_name+'_numClasses7')
