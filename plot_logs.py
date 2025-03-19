import os 
import sys 
import matplotlib.pyplot as plt

def load_log_file(filepath):
    keywords = {"train_loss_con": [], "Test Accuracy": [], "train_loss_sup": []}
    with open(filepath, 'r') as f:
        data = list(map(lambda x: x.replace("\n", ""), f.readlines()))
        for line in data:
            split_line = line.split(":")
            for keyword in keywords.keys():
                if keyword in line:
                    keywords[keyword].append(split_line[-1])

    return keywords

def plot_loss_test_accuracy(data_dir):
    all_files = list(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))
    all_files_data = {}
    keywords_found = 0
    keywords = {}

    for files in all_files:
        all_files_data[files.split('/')[-1]] = load_log_file(files)
        if not keywords_found:
            keywords = list(all_files_data[files.split('/')[-1]].keys())
            keywords_found = 1

    print(keywords)
    print(all_files_data.keys())
    
    for keyword in keywords:
        for filename, data in all_files_data.items():
            plt.plot(list(range(1,len(data[keyword]) + 1)), data[keyword], label = f"{filename}_{keyword}")
        plt.xlabel("Iterations")
        plt.ylabel(keyword)
        plt.title(f"{keyword} vs Iterations")
        plt.legend()
        plt.savefig(f"plots/{keyword}.png")
        # plt.show()
    

if __name__ == "__main__":
    log_dir = sys.argv[1]

    print(plot_loss_test_accuracy(log_dir))