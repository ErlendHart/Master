import pydicom
import uuid


def generate_uid():
    # This is an example root UID; replace it with your institution's root UID if you have one
    # base_uid = "1.2.826.0.1.3680043.2.135.1066.101"
    base_uid = "1.2.246.352.205.5367972663356174645"
    # Generates a random 12-digit number
    unique_id = str(uuid.uuid4().int)[:12]
    new_uid = f"{base_uid}.{unique_id}"
    return new_uid


new_sop_instance_uid = generate_uid()
new_series_instance_uid = generate_uid()

print(new_sop_instance_uid)
print(new_series_instance_uid)


# Generate new UIDs
new_sop_instance_uid = generate_uid()
new_series_instance_uid = generate_uid()


# Define the file paths and the new Series Instance UID

original_dicom_path = input("Enter the path to the DICOM file: ").strip()

# Replace with the actual path to your DICOM file
# original_dicom_path = "/Home/siv32/cav015/Software/Annet/Fluka/New_CTV/sb17/RS.1.2.246.352.205.5367972663356174645.3688463234413044619.dcm"
# Replace with the desired path to save the modified file
# modified_dicom_path = "/Home/siv32/cav015/Software/Annet/Fluka/New_CTV/sb36/RS.1.2.246.352.205.5580736609512395861.4898981830543392146.dcm"
modified_dicom_path = original_dicom_path

# The new Series Instance UID
# new_studyID = "Erlend123"

# The new Series Instance UID
# new_StructureSetLabel = "Erlend1234"

new_Series_Instance_UID = "1.2.246.352.205.4861758185314356046.4665645410583427999"


# new_Media_Storage_SOP_Instance_UID = "1.2.246.352.205.5367972663356174645.1111111111111111111"

# Load the DICOM file
ds = pydicom.dcmread(original_dicom_path)

# Change the Series ID
# ds.StudyID = new_studyID

# Change the Series Instance UID
# ds.StructureSetLabel = new_StructureSetLabel

# ds.SeriesInstanceUID = new_Series_Instance_UID
ds.file_meta.MediaStorageSOPInstanceUID = new_sop_instance_uid
ds.SOPInstanceUID = new_sop_instance_uid
ds.SeriesInstanceUID = new_series_instance_uid

# ds.SOPInstanceUID = new_Media_Storage_SOP_Instance_UID
# ds.MediaStorageSOPInstanceUID = new_Media_Storage_SOP_Instance_UID

# Save the modified file
ds.save_as(modified_dicom_path)
