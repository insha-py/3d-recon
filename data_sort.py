import os

folder_path = r"C:\Users\Insha\Pictures\ashoka reconstruction\000001"

# Get all image files (you can adjust extensions)
files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Sort based on original filename (your format already sorts correctly lexicographically)
files.sort()

# Rename with zero padding
for idx, filename in enumerate(files, start=1):
    ext = os.path.splitext(filename)[1]
    new_name = f"{idx:06d}{ext}"   # 000001.jpg format
    
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    
    os.rename(src, dst)

print("Renaming complete!")