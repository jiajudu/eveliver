import os
import shutil


def clean():
    os.chdir('r')
    for d in os.listdir():
        if not os.path.exists(os.path.join(d, 'output', 'f.json')):
            shutil.rmtree(d)


if __name__ == '__main__':
    clean()
