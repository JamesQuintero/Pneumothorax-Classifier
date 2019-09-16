import glob #for loading DICOM files from disk
import pydicom #for reading DICOM files
import matplotlib.pyplot as plt #for displaying DICOM files
import pandas as pd #for reading data files

from mask_functions import rle2mask


images_path = './data/stage_2_images/*.dcm'

def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


#plots many of the DICOM images at once
def plot_many_images():
    start = 5   # Starting index of images
    num_img = 4 # Total number of images to show

    fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
    dataset = glob.glob(images_path)

    print("Len dataset: "+str(len(dataset)))

    for q, file_path in enumerate(dataset[start:start+num_img]):

        print(str(q)+": "+str(file_path))
        dataset = pydicom.dcmread(file_path)
        #show_dcm_info(dataset)
        
        ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)


#adds mask to provided plot
# def add_mask():


def display_masks():
    start = 0   # Starting index of images
    num_img = 5 # Total number of images to show

    images_path = "./data/sample_train/*.dcm"
    training_labels_path = "./data/train_labels.csv"


    df = pd.read_csv(training_labels_path)

    #prints the first 5 rows of each column separately
    # print(df.head(5))

    print("Num training labels: "+str(len(df)))

    #creates charts/images
    fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))

    images = glob.glob(images_path)

    for q, file_path in enumerate(images[start:start+num_img]):
        dataset = pydicom.dcmread(file_path)

        # print(str(q)+": "+str(file_path))

        #displays the body scan
        ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)

        #extracts image_id from the file path
        image_id = file_path.split('\\')[-1].replace(".dcm", "")
        print("Imageid: "+str(image_id))

        #finds masks matching image_id
        found_masks = df[df['ImageId'].str.match(image_id)]
        found_masks = found_masks.values.tolist()


        

        #has multiple masks
        if len(found_masks)>1:
            print("Multiple masks")

            # print(mask_data)

            for x in range(0, len(found_masks)):
                mask = rle2mask(found_masks[x][1], 1024, 1024).T

            ax[q].set_title('See Marker')
            ax[q].imshow(mask, alpha=0.3, cmap="Reds")

        #has single mask
        elif len(found_masks)==1 and found_masks[0][1] != '-1':
            print("Single mask")

            mask_data = found_masks[0][1]
            # print("Mask data: "+str(mask_data))

            mask = rle2mask(mask_data, 1024, 1024).T
            ax[q].set_title('See Marker')
            ax[q].imshow(mask, alpha=0.3, cmap="Reds")

        #has no masks
        else:
            print("Nothing to see")
            ax[q].set_title('Nothing to see')




        print()

    #displays everything
    plt.show()



# for file_path in glob.glob(images_path):
#     dataset = pydicom.dcmread(file_path)
#     show_dcm_info(dataset)
#     plot_pixel_array(dataset)
#     # break # Comment this out to see all

# plot_many_images()
display_masks()