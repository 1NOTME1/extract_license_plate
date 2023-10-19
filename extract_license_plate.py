import os
import shutil
import cv2
import pytesseract
from PIL import Image

class LicensePlateExtractor:
    def __init__(self, txt_file_path, destination_folder):
        self.txt_file_path = txt_file_path
        self.destination_folder = destination_folder

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

    def extract_license_plate(self, image_path):
        print(f"Przetwarzanie obrazu: {image_path}")
        
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Debugowanie: Wyświetlanie obrazu w odcieniach szarości
        cv2.imshow('Odcienie szarości', gray_image)
        cv2.waitKey(0)
        
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Debugowanie: Wyświetlanie rozmytego obrazu
        cv2.imshow('Rozmycie', blurred_image)
        cv2.waitKey(0)
        
        edged_image = cv2.Canny(blurred_image, 100, 300)
        
        # Debugowanie: Wyświetlanie obrazu z wykrytymi krawędziami
        cv2.imshow('Krawędzie', edged_image)
        cv2.waitKey(0)
        
        contours, _ = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                license_plate = gray_image[y:y+h, x:x+w]
                
                # Debugowanie: Wyświetlanie wykrytej tablicy rejestracyjnej
                cv2.imshow('Tablica rejestracyjna', license_plate)
                cv2.waitKey(0)
                
                break
        else:
            print("Nie znaleziono tablicy rejestracyjnej.")
            return None

        _, threshold_image = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        larger_image = cv2.resize(threshold_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        custom_oem_psm_config = r'--oem 3 --psm 7'
        license_plate_text = pytesseract.image_to_string(Image.fromarray(larger_image), config=custom_oem_psm_config).strip()

        return license_plate_text

    def process_files(self):
        with open(self.txt_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.endswith(('.jpg', '.png')):
                license_plate_text = self.extract_license_plate(line)
                if license_plate_text:
                    safe_text = ''.join(e for e in license_plate_text if e.isalnum())
                    new_file_path = os.path.join(self.destination_folder, f"{safe_text[:50]}.{line.split('.')[-1]}")
                    shutil.copy(line, new_file_path)

if __name__ == "__main__":
    txt_file_path = 'sciezki.txt'
    destination_folder = 'new'
    extractor = LicensePlateExtractor(txt_file_path, destination_folder)
    extractor.process_files()
