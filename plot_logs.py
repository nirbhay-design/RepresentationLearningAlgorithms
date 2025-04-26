import os 
import sys 
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

def load_log_file(filepath):
    keywords = {"train_loss_con": [], "Test Accuracy": [], "train_loss_sup": []}
    with open(filepath, 'r') as f:
        data = list(map(lambda x: x.replace("\n", ""), f.readlines()))
        for line in data:
            split_line = line.split(":")
            for keyword in keywords.keys():
                if keyword in line:
                    keywords[keyword].append(float(split_line[-1]))

    return keywords

def plot_loss_test_accuracy(data_dir):
    filenames = os.listdir(data_dir)
    filenames.remove("dare.c10.log")
    all_files = list(map(lambda x: os.path.join(data_dir, x), filenames))
    all_files_data = {}
    keywords_found = 0
    keywords = {}

    for files in all_files:
        all_files_data[files.split('/')[-1]] = load_log_file(files)
        if not keywords_found:
            keywords = list(all_files_data[files.split('/')[-1]].keys())
            keywords_found = 1

    datanames = {'c10': [], 'c100': []}

    for file in all_files_data.keys():
        algo, data = file.split(".")[:2]
        print(f"[{file}] Best Test Accuracy: algo: {algo}, dataset: {data}: {max(all_files_data[file]['Test Accuracy'])}")
        datanames[data].append(file) 

#     for data, datafiles in datanames.items():
#         k1 = "Test Accuracy"
#         plot_based_on_keywords(data, k1, [all_files_data[i][k1] for i in datafiles], datafiles)
            

# def plot_based_on_keywords(dataname, keyword, keyword_data, filenames):
#     plt.figure()
#     for data, filename in zip(keyword_data, filenames):
#         plt.plot(list(range(1,len(data) + 1)), data, label = f"{filename}")
#     plt.xlabel("Iterations")
#     plt.ylabel(keyword)
#     plt.title(f"{keyword} vs Iterations for {dataname}")
#     plt.legend()
#     plt.savefig(f"plots/{keyword}_{dataname}.png")

if __name__ == "__main__":
    log_dir = sys.argv[1]

    plot_loss_test_accuracy(log_dir)