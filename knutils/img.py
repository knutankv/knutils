from pathlib import Path


def convert_heic(files):
    """Convert HEIC files to JPG. Requires pillow-heif."""
    from PIL import Image
    import pillow_heif

    pillow_heif.register_heif_opener()

    for f in files:
        src = Path(f)
        dst = src.with_suffix('.jpg')
        img = Image.open(src)
        img.save(dst, 'JPEG')
        print(f'{src} -> {dst}')


def convert_mov(files):
    """Convert MOV files to MP4 using ffmpeg."""
    import subprocess

    for f in files:
        src = Path(f)
        dst = src.with_suffix('.mp4')
        subprocess.run(
            ['ffmpeg', '-i', str(src), '-vcodec', 'copy', '-acodec', 'copy', str(dst)],
            check=True,
        )
        print(f'{src} -> {dst}')
