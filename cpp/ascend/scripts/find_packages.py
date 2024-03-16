import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)

    parser.add_argument("-ip", "--host", type=str)
    parser.add_argument("-u", "--user", type=str)
    parser.add_argument("-p", "--port", type=int, default=22)

    parser.add_argument("-l", "--location", type=str)
    
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    filename = os.path.abspath(args.file)
    command = f"ldd {filename}"
    
    files=[]
    for line in os.popen(command).read().split("\n"):
        if "not found" in line:
            lib_name = line.split()[0]
            files.append(lib_name)
    
    command = "scp "
    for f in files:
        command += f"{f} "
    command += f"{args.user}@{args.host}:{args.location}" + ("" if args.port == 22 else f" -p {args.port}")

    print(command)
         

            
    
