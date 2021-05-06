import matplotlib.pyplot as plt
import numpy as np

# runtime of batch size 32 is 134546ms for 9600 images
# runtime of batch size 64 is 123315ms for 9600 images
# runtime of batch size 128 is 119577ms for 9600 images
# single 32 384172ms 9600 images
# single 64 374289ms 9600 images
# single 128 370257ms 9600 images
# tf 0.9936 accuracy batch size 32 5 epochs 89.80s
# tf 0.9924 accuracy batch size 64 5 epochs 84.94s
# tf 0.9906 accuracy batch size 128 5 epochs 79.52s
full_path = '/Users/liam_adams/my_repos/csc724_project/'

def plot_runtime(batch_size):
    grad_calc1_fn = 'gc8080_runtime_metrics/grad_calc8080_{}.txt'.format(batch_size)
    grad_calc2_fn = 'gc8081_runtime_metrics/grad_calc8081_{}.txt'.format(batch_size)
    grad_calc3_fn = 'gc8082_runtime_metrics/grad_calc8082_{}.txt'.format(batch_size)
    grad_calc4_fn = 'gc8083_runtime_metrics/grad_calc8083_{}.txt'.format(batch_size)
    optimizer_fn = 'gc8080_runtime_metrics/optimizer_{}.txt'.format(batch_size)
    
    with open(full_path + grad_calc1_fn) as f:
        grad_calc1_content = f.readlines()
    grad_calc1_content = [x.strip() for x in grad_calc1_content]

    with open(full_path + grad_calc1_fn) as f:
        grad_calc2_content = f.readlines()
    grad_calc2_content = [x.strip() for x in grad_calc2_content] 

    with open(full_path + grad_calc1_fn) as f:
        grad_calc3_content = f.readlines()
    grad_calc3_content = [x.strip() for x in grad_calc3_content] 

    with open(full_path + grad_calc1_fn) as f:
        grad_calc4_content = f.readlines()
    grad_calc4_content = [x.strip() for x in grad_calc4_content] 

    with open(full_path + grad_calc1_fn) as f:
        optimizer_content = f.readlines()
    optimizer_content = [x.strip() for x in optimizer_content]


    grad_calc_np = np.asarray(grad_calc1_content, dtype=np.float32)
    optimizer_np = np.asarray(optimizer_content, dtype=np.float32)
    total_runtime = grad_calc_np + optimizer_np
    plt.plot(total_runtime)
    plt.ylim(ymin=0)
    plt.ylabel('time (ms)')
    plt.title('Runtime of batch size ' + str(batch_size))
    plt.show()

def plot_memory(batch_size):
    grad_calc1_fn = 'gc8080_runtime_metrics/memory_{}.txt'.format(batch_size)
    with open(full_path + grad_calc1_fn) as f:
        grad_calc1_content = f.readlines()
    grad_calc1_content = [x.strip() for x in grad_calc1_content]
    grad_calc1_content = [x[:-3] for x in grad_calc1_content]

    grad_calc_np = np.asarray(grad_calc1_content, dtype=np.float32)
    plt.plot(grad_calc_np)
    plt.ylim(ymin=0)
    plt.ylabel('Memory (MiB)')
    plt.title('Memory usage for batch size ' + str(batch_size))
    plt.show()

def plot_opt_memory():
    grad_calc1_fn = 'opt_runtime_metrics/memory.txt'
    with open(full_path + grad_calc1_fn) as f:
        grad_calc1_content = f.readlines()
    grad_calc1_content = [x.strip() for x in grad_calc1_content]
    grad_calc1_content = [x[:-3] for x in grad_calc1_content]

    grad_calc_np = np.asarray(grad_calc1_content, dtype=np.float32)
    plt.plot(grad_calc_np)
    plt.ylim(ymin=0)
    plt.ylabel('Memory (MiB)')
    plt.title('Memory usage for optimizer')
    plt.show()

def plot_single_runtime(batch_size):
    grad_calc1_fn = 'single_runtime_metrics/single_{}.txt'.format(batch_size)
    with open(full_path + grad_calc1_fn) as f:
        grad_calc1_content = f.readlines()
    grad_calc1_content = [x.strip() for x in grad_calc1_content]

    grad_calc_np = np.asarray(grad_calc1_content, dtype=np.float32)
    plt.plot(grad_calc_np)
    plt.ylim(ymin=0)
    plt.ylabel('Time (ms)')
    plt.title('Single core runtime of batch size ' + str(batch_size))
    plt.show()


if __name__ == '__main__':
    #plot_runtime(128)
    #plot_memory(128)
    #plot_opt_memory()
    plot_single_runtime(128)