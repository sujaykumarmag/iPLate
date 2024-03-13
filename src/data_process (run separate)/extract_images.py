#######################################################################################################
# Author : SujayKumar Reddy M
# Project : iPLate 
# Description : Making Augmented Dataset for the Use of model
# Sharing : Hemanth Karnati, Melvin Paulsam
# School : Vellore Institute of Technology, Vellore
# Project Manager : Prof. Dr. Swarnalatha P
#######################################################################################################


import fitz  

def extract_images_from_pdf(pdf_file):
    images = []
    pdf_document = fitz.open(pdf_file)
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    pdf_document.close()
    return images

if __name__ == "__main__":
    pdf_file_path = "../../data/doc4.pdf"  # Change this to the path of your PDF file
    extracted_images = extract_images_from_pdf(pdf_file_path)
    for i, image_data in enumerate(extracted_images, start=1):
        with open(f"image_{i}.png", "wb") as img_file:
            img_file.write(image_data)
        print(f"Image {i} extracted successfully.")
