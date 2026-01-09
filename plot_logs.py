import os 
import sys 
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

def load_log_file(filepath):
    keywords = {"train_loss_con": [], "Test Accuracy": [], "train_loss_sup": [], "knn_acc": [], "log_reg_acc": []}
    with open(filepath, 'r') as f:
        data = list(map(lambda x: x.replace("\n", ""), f.readlines()))
        for line in data:
            split_line = line.split(":")
            for keyword in keywords.keys():
                if keyword in line:
                    keywords[keyword].append(float(split_line[-1]))

    if len(keywords["Test Accuracy"]) == 0:
        raise Exception("This is issue")

    return keywords

def plot_loss_test_accuracy(data_dir):
    filenames = os.listdir(data_dir)
    all_files = list(map(lambda x: os.path.join(data_dir, x), filenames))
    all_files_data = {}
    keywords_found = 0
    keywords = {}

    for files in all_files:
        try:
            all_files_data[files.split('/')[-1]] = load_log_file(files)
            if not keywords_found:
                keywords = list(all_files_data[files.split('/')[-1]].keys())
                keywords_found = 1
        except:
            print(f"{files} has some issue")

    
    algo_datas = {}

    for file in all_files_data.keys():
        algo, data, model = file.split(".")[:3]
        if model == "log":
            model = "r50"
        combine = f"{data}.{model}"
        print(f"[{file}] Best Test Accuracy: algo: {algo}, dataset: {data}: {max(all_files_data[file]['Test Accuracy'])}", end = "")
        if len(all_files_data[file]["knn_acc"]) != 0:
            print(f", knn_acc: {all_files_data[file]['knn_acc'][0]}, log_reg_acc: {all_files_data[file]['log_reg_acc'][0]}")
        else:
            print()
        if algo not in algo_datas:
            algo_datas[algo] = {}
        algo_datas[algo][combine] = all_files_data[file]['train_loss_con'] 

    for algo, algo_data in algo_datas.items():
        plot_based_on_keywords(algo, algo_data)
            
def plot_based_on_keywords(algo, algo_data):
    plt.figure()

    # marker = ['^', 'v', '*', 'D']
    # marker_map = dict(zip(algo_data.keys(), marker))
    for keyword, data in algo_data.items():
        plt.plot(list(range(1,len(data) + 1)), data, label = f"{keyword}")
        # plt.plot(list(range(1,len(data) + 1)), data, label = f"{keyword}", marker='o', markersize=3)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xlabel("Iterations")
    plt.ylabel("Contrastive Loss")
    plt.title(f"Loss vs Iterations for {algo}")
    plt.legend()
    plt.savefig(f"loss_plots/{algo}.png")
    plt.close()

if __name__ == "__main__":
    log_dir = sys.argv[1]

    plot_loss_test_accuracy(log_dir)