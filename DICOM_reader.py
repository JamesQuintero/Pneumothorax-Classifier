import glob #for loading DICOM files from disk
import pydicom #for reading DICOM files
import matplotlib.pyplot as plt #for displaying DICOM files


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


def display_masks():
    df = pd.read_csv('./data/train_labels.csv', header=None, index_col=0)

    print("Num training labels: "+str(df))

    fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
    for q, file_path in enumerate(glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')[start:start+num_img]):
        dataset = pydicom.dcmread(file_path)
        #print(file_path.split('/')[-1][:-4])
        ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)
        if df.loc[file_path.split('/')[-1][:-4],1] != '-1':
            mask = rle2mask(df.loc[file_path.split('/')[-1][:-4],1], 1024, 1024).T
            ax[q].set_title('See Marker')
            ax[q].imshow(mask, alpha=0.3, cmap="Reds")
        else:
            ax[q].set_title('Nothing to see')


# for file_path in glob.glob(images_path):
#     dataset = pydicom.dcmread(file_path)
#     show_dcm_info(dataset)
#     plot_pixel_array(dataset)
#     # break # Comment this out to see all

# plot_many_images()
display_masks()