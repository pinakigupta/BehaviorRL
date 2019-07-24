import os, sys, glob, time
from mpi4py import MPI
from settings import req_dirs, models_folder, ray_folder
from shutil import copyfile

pathname = os.getcwd()
homepath = os.path.expanduser("~")
s3pathname = homepath+'/s3-drive/groups/Behavior/Pinaki'


def makedirpath(pathname=None):
    if pathname is None:
        print("No path provided for creation.")
        return
    if not os.path.exists(pathname):
        try:
            os.makedirs(pathname, exist_ok=True)
        except:
            print("Failed to created s3 drive at", pathname)
            pass

first_default_args_call = True
LOAD_PREV_MODEL = True
RUN_WITH_RAY = True

###############################################################
#        DEFINE YOUR "BASELINE" (AGENT) PARAMETERS HERE
###############################################################
'''
train_env_id = 'parking_2outs-v0'
play_env_id = 'parking_2outs-v0' 
alg = 'her'
network = 'mlp'
num_timesteps = '1' # Keeping steps at 1 will only sping off prediction/simulation. > 1 for training. 
# To be compatible with Ray please keep this a normal integer representation. i.e 1000 not 1e3

'''
train_env_id = 'two-way-v0'
play_env_id = 'two-way-v0'
alg = 'ppo2'
network = 'mlp'
num_timesteps = '1' # Keeping steps at 1 will only sping off prediction/simulation. > 1 for training. 
# To be compatible with Ray please keep this a normal integer representation. i.e 1000 not 1e3

#################################################################


urban_AD_env_path = pathname + '/urban_env/envs'
sys.path.append(urban_AD_env_path)
import site
site.addsitedir(urban_AD_env_path)





def create_dirs(req_dirs):
    for dirName in req_dirs:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            # print("Directory " , dirName ,  " Created ")
        # else:
            # print("Directory " , dirName ,  " already exists")


def is_master():
    return MPI.COMM_WORLD.Get_rank() == 0




if is_master():
    InceptcurrentDT = time.strftime("%Y%m%d-%H%M%S")
else:
    InceptcurrentDT = None

InceptcurrentDT = MPI.COMM_WORLD.bcast(InceptcurrentDT, root=0)


def is_predict_only():
    return float(num_timesteps) == 1





def default_args(save_in_sub_folder=None):
    create_dirs(req_dirs)

    currentDT = time.strftime("%Y%m%d-%H%M%S")
    global first_default_args_call
    ####################################################################
    # DEFINE YOUR SAVE FILE, LOAD FILE AND LOGGING FILE PARAMETERS HERE
    ####################################################################
    modelpath = pathname
    if RUN_WITH_RAY:
        modelpath += '/' + ray_folder 
    '''try:
        if os.path.exists(s3pathname):
            modelpath = s3pathname
    except:
        print("s3 pathname doesn't exist")'''

    save_folder = modelpath + '/' + models_folder + \
        '/' + train_env_id + '/' + alg + '/' + network
    load_folder = modelpath + '/' + models_folder + \
        '/' + train_env_id + '/' + alg + '/' + network

    if first_default_args_call:
        list_of_file = glob.glob(load_folder+'/*')
        if save_in_sub_folder is not None:
            save_folder += '/' + str(save_in_sub_folder)
        save_file = save_folder + '/' + str(currentDT)
        first_default_args_call_Trigger = True
    else:
        if save_in_sub_folder is not None:
            save_folder += '/' + str(save_in_sub_folder)
        save_file = save_folder + '/' + str(currentDT)
        list_of_file = glob.glob(save_folder+'/*')
        first_default_args_call_Trigger = False

    # Specifiy log directories for open AI
    '''logger_path = save_folder + '/log/'
    tb_logger_path = save_folder + '/tb/'
    os.environ['OPENAI_LOGDIR'] = logger_path
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'''

    ###############################################################

    DEFAULT_ARGUMENTS = [
        '--env=' + train_env_id,
        '--alg=' + alg,
        '--network=' + network,
        '--num_timesteps=' + num_timesteps,
        #    '--num_env=0',
        #    '--save_path=' + save_file,
        #     '--tensorboard --logdir=' + tb_logger_path,
        #    '--play'
        #    '--num_env=8'
    ]

    DEFAULT_ARGUMENTS_DICT = {
        'env': train_env_id,
        'alg': alg,
        'network': network,
        'num_timesteps': num_timesteps
    }

    def copy_terminal_output_file():
        src = os.getcwd() + '/' + terminal_output_file_name
        dst = save_folder + '/' + terminal_output_file_name
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if os.path.exists(src):
            copyfile(src, dst)
        else:
            print("out put file ",terminal_output_file_name,"doesn't exist")

    def create_save_folder(save_folder):
        try:
            os.makedirs(save_folder)
        except OSError:
            # print ("Creation of the save path %s failed. It might already exist" % save_folder)
            a = 1
        else:
            print("Successfully created the save path folder %s " % save_folder)

    def save_model(save_file=None):
        if save_file is not None:
            if not is_predict_only():
                if MPI is None or is_master():
                    create_save_folder(save_folder=save_folder)
                    DEFAULT_ARGUMENTS.append('--save_path=' + save_file)
                    DEFAULT_ARGUMENTS_DICT['save_path'] = save_file
                    print("Saving file", save_file)
                    copy_terminal_output_file()
                    # DEFAULT_ARGUMENTS.append('--tensorboard --logdir=' + tb_logger_path)
        return

    def load_model(load_file=None):
        if load_file is not None:
            if (not LOAD_PREV_MODEL) and first_default_args_call:
                return
            DEFAULT_ARGUMENTS.append('--load_path=' + load_file)
            DEFAULT_ARGUMENTS_DICT['load_path'] = load_file
            print("Loading file", load_file)
        return

    terminal_output_file_name = 'output.txt'

    def is_empty_directory(directorypath):
        if not os.path.isdir(directorypath):
            return False
        if not os.listdir(directorypath):
            return True
        return False

    def filetonum(filename):
        try:
            return int(filename.split('/')[-1].replace('-', ''))
        except:
            return -1

    def purge_names_not_matching_pattern(list_of_file_or_folders):
        if not list_of_file_or_folders:
            return None
        for fileorfoldername in list_of_file_or_folders:
            if '.' in fileorfoldername:
                list_of_file_or_folders.remove(fileorfoldername)
            # remove empty directories
            elif is_empty_directory(directorypath=fileorfoldername):
                list_of_file_or_folders.remove(fileorfoldername)
        return list_of_file_or_folders

    def latest_model_file_from_list_of_files_and_folders(list_of_files):
        list_of_file_or_folders = purge_names_not_matching_pattern(
            list_of_file_or_folders=list_of_files)
        if not list_of_file_or_folders:
            return None
        latest_file_or_folder = max(list_of_file_or_folders, key=filetonum)
        if os.path.isdir(latest_file_or_folder):
            list_of_files_and_folders_in_subdir = glob.glob(
                latest_file_or_folder+'/*')
            latest_model_file_in_subdir = \
                latest_model_file_from_list_of_files_and_folders(
                    list_of_files_and_folders_in_subdir)
            if latest_model_file_in_subdir is None:
                list_of_file_or_folders.remove(latest_file_or_folder)
                return latest_model_file_from_list_of_files_and_folders(list_of_file_or_folders)
            else:
                return latest_model_file_in_subdir
        return latest_file_or_folder  # must be a file

    if list_of_file:  # is there anything in the save directory
        if save_in_sub_folder is None:
            load_last_model = LOAD_PREV_MODEL
        else:
            load_last_model = LOAD_PREV_MODEL or not first_default_args_call

        if load_last_model:
            latest_file = latest_model_file_from_list_of_files_and_folders(
                list_of_files=list_of_file)
            load_model(load_file=latest_file)
            save_model(save_file=save_file)
        else:
            save_model(save_file=save_file)
    else:
        print(" list_of_file empty in load path ", load_folder)
        save_model(save_file=save_file)

    # print(" DEFAULT_ARGUMENTS ", DEFAULT_ARGUMENTS)

    if first_default_args_call_Trigger:
        first_default_args_call = False

    return DEFAULT_ARGUMENTS, DEFAULT_ARGUMENTS_DICT

