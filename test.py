"""from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
uri = ''
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = (r'{}data/input/img_0.jpeg').format(uri)
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)


# draw result
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path=(r'{}data/input/simfang.ttf').format(uri))
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')"""

"""import cv2
import pytesseract
import pandas as pd

uri =''
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = (r"{}model/Tesseract-OCR/tesseract.exe").format(uri)

image = cv2.imread((r'{}data/input/img_0.jpeg').format(uri))

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


image1 = get_grayscale(image)
image1 = thresholding(image)    
#thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

x,y,w,h = 37, 625, 309, 28  
ROI = image1[y:y+h,x:x+w]
data = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6')
print(data)

cv2.imshow('thresh', image1)
cv2.imshow('ROI', ROI)
cv2.waitKey()"""

import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
from PIL import Image, ImageEnhance
from paddleocr import PaddleOCR, draw_ocr
import math  

#img = cv2.imread("img_0.jpeg")
#fp = Path((r'{}data/input/img_0.jpeg').format(uri))
#img_path = 'img_0.jpeg'
uri = ''



ocr = PaddleOCR(lang='ch')
result = ocr.ocr((r'{}data/input/img_2.jpeg').format(uri), cls=False)
column = ["Date", "Description", "Deposits", "Withdrawals", "Balance"]
lines = []
restxt = []
datacoordx= {}
datacoordy= {}

#draw result
"""image = Image.open((r'{}data/input/img_0.jpeg').format(uri)).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path=(r'{}data/input/simfang.ttf').format(uri))
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')"""


#print(result)

for line in result:
    #restxt.append(line[0][0])
    if line[1][0][:4] == 'Date':
        datacoordx['Date'] = line[0][0][0] #, line[0][2]
        datacoordy['Date'] = line[0][3][0]
    elif line[1][0][:11] == 'Description':
        datacoordx['Description'] = line[0][0][0] #, line[0][2]
        datacoordy['Description'] = line[0][3][0]
    elif line[1][0][:8] == 'Deposits': #or line[1][0] == 'Deposits ($]' or line[1][0] == 'Deposits($)':
        datacoordx['Deposits'] = line[0][1][0] #, line[0][2]
        datacoordy['Deposits'] = line[0][2][0]
    elif line[1][0][:11] == 'Withdrawals': #or line[1][0] == 'Withdrawals (S)' or line[1][0] == 'Withdrawals ($]' or line[1][0] == 'Withdrawals($)':
        datacoordx['Withdrawals'] = line[0][1][0] #, line[0][2]
        datacoordy['Withdrawals'] = line[0][2][0]
    elif line[1][0][:7].lower() == 'balance': #or line[1][0] == 'Balance ($)' or line[1][0] == 'Balance (S':
        datacoordx['Balance'] = line[0][1][0] #, line[0][2]
        datacoordy['Balance'] = line[0][2][0]
    else:
        pass
 
#print(datacoordx)
 
"""datacoordx['Date'] = [line[0][0][0] for line in result if 'Date' in str(line[1][0])]
datacoordx['Description'] = [line[0][0][0] for line in result if 'Description' in str(line[1][0])]
datacoordx['Deposits'] = [line[0][1][0] for line in result if 'Deposits ($)' in str(line[1][0])]
datacoordx['Withdrawals'] = [line[0][1][0] for line in result if 'Withdrawals C$]' in str(line[1][0])]
datacoordx['Balance'] = [line[0][1][0] for line in result if 'BALANCE ($)' in str(line[1][0])]"""
 
Date = [] 
Description = []
Deposits = []
Withdrawals = []
Balance = []
fresult = {}
Desc1 = Desc2 = Desc3 = Desc4 = []
coord = '' 

Date = [[line[1][0], line[0][1]] for line in result if math.isclose(line[0][0][0], datacoordx['Date'], abs_tol = 20.0) and line[1][0] != 'Date']
#print(Date)
Desc1 = [[line[0][0], line[0][1], line[1][0]] for line in result if math.isclose(line[0][0][0], datacoordx['Description'], abs_tol = 20.0) and line[1][0] != 'Description']
Desc2 = [[line[0][0], line[0][1], line[1][0]] for line in result if math.isclose(line[0][0][0], datacoordx['Description'], abs_tol = 60.0) and line[1][0] != 'Description']
#print(Desc1)
#print(Desc2)
Desc3 = [ x for x in Desc2 if not x in Desc1 ]
#print(Desc3)
for dt in Desc1:
    for dt1 in Desc3:
        if math.isclose(dt1[0][0], dt[0][0], abs_tol = 70.0) and math.isclose(dt1[0][1], dt[0][1], abs_tol = 70.0) and dt1[0][1] >= dt[0][1] and dt[2]!=dt1[2] and dt[2]!= 'Opening Balance':
            Description.append([dt[2] +' '+ dt1[2], dt1[1]])
            coord = dt[1]
    else:
        if coord != dt[1] and dt[2]!= 'Opening Balance':
            Description.append([dt[2], dt[1]])
  
#print(Description)

Deposits = [[line[1][0], line[0][1]] for line in result if math.isclose(line[0][1][0], datacoordx['Deposits'], abs_tol = 20.0) and line[1][0] != 'Deposits ($)']
Withdrawals = [[line[1][0], line[0][1]] for line in result if math.isclose(line[0][1][0], datacoordx['Withdrawals'], abs_tol = 20.0) and line[1][0] != 'Withdrawals C$]']
Balance = [[line[1][0], line[0][1]] for line in result if math.isclose(line[0][1][0], datacoordx['Balance'], abs_tol = 20.0) and line[1][0] != 'BALANCE ($)']
#print(Withdrawals)


import math  
data = {}
lstdt = ''
for idx,d in enumerate(Description):
    for dt in Date:
	    if math.isclose(d[1][1], dt[1][1], abs_tol = 30.0):
	        lstdt = dt[0]
	        data[idx] = [dt[0],d[0]]
    else:
        data[idx] = [lstdt,d[0]]
        
    for wdrl in Withdrawals:
        if math.isclose(d[1][1], wdrl[1][1], abs_tol = 50.0):
            data[idx].append(wdrl[0])
    else:
        data[idx].append('')
        
    for dpt in Deposits:
        if math.isclose(d[1][1], dpt[1][1], abs_tol = 20.0):
            data[idx].append(dpt[0])
            break
    """else:
        data[idx].append('')"""
        
    for bln in Balance:
        if math.isclose(d[1][1], bln[1][1], abs_tol = 50.0):
            data[idx].append(bln[0])
            break
    else:
        data[idx].append('')
#print(data)

import json

json_object = json.dumps(data, indent = 4) 
#print(json_object)

#data1 = json.loads(data)
#print(data1)


data_frame = pd.DataFrame(json_object, columns=['Date', 'Description', 'Withdrawals', 'Deposits', 'Balance'])
print(data_frame)
"""for line in result:   
    if math.isclose(line[0][0][0], datacoordx['Date'], abs_tol = 10.0):
        response['Date'] = line[1][0]
    else:
        response['Date'] = ''
        
    if math.isclose(line[0][0][0], datacoordx['Description'], abs_tol = 10.0):
        response['Description'] = line[1][0]
    else:
        response['Description'] = ''
        
    if math.isclose(line[0][1][0], datacoordx['Deposits'], abs_tol = 10.0):
        response['Deposits'] = line[1][0]
    else:
        response['Deposits'] = ''
        
    if math.isclose(line[0][1][0], datacoordx['Withdrawals'], abs_tol = 10.0):
        response['Withdrawals'] = line[1][0]
    else: 
        response['Withdrawals'] = ''
        
    if math.isclose(line[0][1][0], datacoordx['Balance'], abs_tol = 10.0):
        response['Balance'] = line[1][0]
    else:
        response['Balance'] = ''"""
        


"""df = pd.DataFrame.from_dict(fresult, orient='index')
mydict = {'Date': ['3 Apr', '4 Apr', '5 Apr', '8 Apr'], 
'Description': ['Opening Balance', 'e-Transfer - Autodeposit', 'RBC MASTERCARD', 'VISATD BANK', 'SUBWAY# 31897', 
'e-Transfer - Autodeposit', 'RBC MASTERCARD', 'VISA TD BANK', 'VISA -CIBC'],
 'Deposits': ['700.00', '204.97', '125.00'], 
 'Withdrawals': ['400.00', '500.00', '500.00', '500.00', '500.00'], 
 'Balance': ['5,575.83', '6,275.83', '5,580.80', '5,528.26', '5,505.00', '5630.0', '5,245.20']}
dict_df = pd.DataFrame({ key:pd.Series(value) for key, value in mydict.items() })
print(dict_df)"""


 
#fresult.update(response)
#print(fresult)
    
    #lines.append(line[1][0])
    #restxt.append(line[1][0])

#tbl_values = restxt

#labels = ['date', 'Description', 'Credit', 'Debit', 'balance']
#df = pd.DataFrame(result, index=labels)   
#print(fresult)

"""custom_config = r'-l eng --oem 3 --psm 6 '

d = pytesseract.image_to_string((r'{}data/input/img_0.jpeg').format(uri), config=custom_config, output_type=Output.DICT)
df = pd.DataFrame(d)

print(df)"""