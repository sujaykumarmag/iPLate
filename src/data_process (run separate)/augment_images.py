#######################################################################################################
# Author : SujayKumar Reddy M
# Project : iPLate 
# Description : Making Augmented Dataset for the Use of model
# Sharing : Hemanth Karnati, Melvin Paulsam
# School : Vellore Institute of Technology, Vellore
# Project Manager : Prof. Dr. Swarnalatha P
#######################################################################################################


# Imports
from serpapi import GoogleSearch
import os, json
import csv
import urllib
import re
import ssl
from os.path import basename
from tqdm import tqdm
from urllib.parse import urlsplit


# PATHS 
SERP_API = "22aae85ff2a6cee9fd48946f44cd02981a19f2b035e2c6848dabbf95a8802888"
EXCTRACTED_DATA = "../../extracted_data/images/"
CSV_DATA = "../../augmented_data/metadata_source/"
ALREADY_ADDED = "../../augmented_data/images"



added = os.listdir(ALREADY_ADDED)
classnames = os.listdir(EXCTRACTED_DATA)
os.makedirs(CSV_DATA,exist_ok=True)
[classnames.remove(i) for i in added]


def get_links(query):
    imgUrls = []
    params = {
    	'api_key': SERP_API,         # your serpapi api
    	'q':query,
    	'engine': 'google_images',    # Serp Api search engine	
    }
    results = GoogleSearch(params).get_dict()['images_results']
    header = ['title', 'thumbnail']
    with open(CSV_DATA+params['q']+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in results:
              writer.writerow([item.get('title'), item.get('thumbnail')])
              imgUrls.append(item.get('thumbnail'))
    return imgUrls



def make_datset(classname, imgUrls):
     IMG_FOLDER = "augmented_data/images/"+classname+"/"
     os.makedirs(IMG_FOLDER,exist_ok=True)
     # download all images
     for imgUrl in imgUrls:
           try:
                 gcontext = ssl._create_unverified_context()
                 imgData = urllib.request.urlopen(imgUrl,context=gcontext).read()
                 fileName = basename(urlsplit(imgUrl)[2]+".png")
                 output = open(IMG_FOLDER+fileName,'wb')
                 output.write(imgData)
                 output.close()
           except:
                  print("Error while request -- Sujay")
                  

for classname in tqdm(classnames):
      # print("--------------------------------------")
      imgUrls = get_links(classname)
      make_datset(classname,imgUrls)
      
     
     


