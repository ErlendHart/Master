# Import all needed libraries
from matplotlib.pyplot import *
from numpy import *
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import glob
from matplotlib.legend import Legend
import pandas as pd
import time
from scipy.optimize import minimize
import scipy
import argparse
import threading
import multiprocessing
# PyQt5 imports
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import PyQt5.Qt
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QButtonGroup, QPushButton, QVBoxLayout, QWidget, QListWidgetItem, QTableWidgetItem
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QVBoxLayout, QGraphicsProxyWidget, QGraphicsPixmapItem
from PyQt5.QtCore import QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import demo
import itertools
import sys
from PyQt5.QtGui import QPixmap
import datetime
import signal
from scipy import optimize
import pydicom as dicom
from scipy.sparse import load_npz, save_npz
import scipy.sparse as sp
import math # - Erlend
# Set parameters for numpy, matplotlib and multiprocessing
multiprocessing.set_start_method("spawn", force=True)
plt.switch_backend('agg')
np.seterr(divide='ignore', invalid='ignore')

lw = 5
lgndsize = 18
txtsize = 30
ticksize = 24


# Defining arguments
parser = argparse.ArgumentParser(epilog="See the wiki for more information at \
https://git.app.uib.no/particletherapyIFT/FRES/wikis/home")

# Arguments for max number of iterations and at what iteration we should plot things
parser.add_argument("-its", "--iterations",
                    help="Maximum number of iterations in the optimizer, default 20", default=20)
parser.add_argument("-its_upd", "--iterations_update",
                    help="How often to plot and update results, default 10", default=10)

# Name of optimization/label
parser.add_argument(
    "-l", "--label", help="Name/label of DVH/LVH-plot and files produced", default="")

# Npz-filename
parser.add_argument(
    "-f", "--npzfile", help="Filename of the npz-file (created using other script)", default="scored_values.npz")

# Opt_parameters file
parser.add_argument("-opt_f", "--opt_param_file",
                    help="Filename of the opt_param_file", default="opt_params.csv")

# Pruning - Not in use
parser.add_argument(
    "-p", "--pruning", help="Filename of pruning file showing which PBs to be pruned", default="")

# Let cutoff values
parser.add_argument("-let_co", "--let_cutoff",
                    help="LET dose cutoff value per fraction, default 0.036", default=0.036)

# Choose alpha-beta values (Needs to be changed to array if multiple values is going to be used)
parser.add_argument("-ab", "--alpha_beta_ratio",
                    help="Alpha beta ratio, default 3.76", default=3.76) # Changed this to 10 - Erlend
# Biological optimization, choose model
parser.add_argument("-bio", "--bio_opt",
                    help="Choose biological model", default="1.1")

# Robust optimization and evaluation, if it should be parallellized as well as how many iterations it should perform.
parser.add_argument("-ro", "--robust_opt", nargs='+',
                    help="Robust optimization if this argument is chosen", default="")
parser.add_argument("-ro_eval", "--robust_eval", nargs='+',
                    help="Choose models to include in the robust evaluation")
parser.add_argument("-para", "--parallell",
                    help="parallelize the robust optimization in this many cores", default=None)
parser.add_argument("-ro_its", "--robust_iterations",
                    help="Number of iterations", default=3)
# Robust biologicla optimaztion and evaluation. Not in use at the moment
parser.add_argument("-ro_bio", "--robust_bio", nargs='+',
                    help="Choose models to include in the biological robust optimization")
parser.add_argument("-ro_bio_eval", "--robust_bio_eval", nargs='+',
                    help="Choose models to include in the biological robust evaluation")
parser.add_argument("-ro_ab", "--robust_ab", nargs='+',
                    help="Choose alpha/beta models to include in the alpha/beta robust optimization", default="")

# Name and location of DICOM-file
parser.add_argument(
    "-dcm", "--dicom", help="Choose alpha/beta models to include in the alpha/beta robust optimization", default="")

# If you want to continue on another optimization, you can choose the .res file here.
parser.add_argument("-c_file", "--continue_file",
                    help="Filename of the continuation file", default="")
parser.add_argument("-eq_weight", "--equal_weighting",
                    help="Initializes the optimizer with all pencilbeams of equal weight", action="store_true")
# Plotting arguments
parser.add_argument("-single", "--plot_single",
                    help="If true, plot only each iteration in the the plots, and not previous", action="store_true")
parser.add_argument("-phys_dose", "--plot_physical_dose",
                    help="If true, plots the physical dose in addition", action="store_true")
parser.add_argument("-frac", "--fractions",
                    help="Number_of_fractions, default 1", default=1)
parser.add_argument("-norm", "--normalize", nargs='+',
                    help="Choose ROI for normalization, and to what dose level the median dose for this ROI should be (seperated by space)", default="")

# Starts the GUI
parser.add_argument(
    "-g", "--gui", help="Run the GUI for the optimizer", action="store_true")
args = parser.parse_args()


# multiplication function
class WorkerThread(QThread):
    """Main optimization class, can be used both through terminal or GUI """
    # Defines the function that sends information back to the GUI
    update_plot = pyqtSignal(list)
    update_dvh = pyqtSignal(list, str)
    update_dvh_ro = pyqtSignal(list, list, list, str, list)
    done_signal = pyqtSignal()

    def __init__(self, mode, path, param_file, DICOM_path, npz_file, number_of_iterations, iterations_update, robust_opt, robust_eval, alpha_beta, robust_bio, bio_opt, let_co, bio_model, single, parallell, robust_iterations, continue_file, plot_physical_dose, opt_param_file, eq_weight, norm):
        """Initalizing the optimizer by either inputs from the GUI, or from arguments, defined at the bottom of the script"""

        # Gets all functions from the GUI
        super().__init__()

        # Initalizes the variables needed in the optimization
        self.path = path
        self.mode = mode
        self.param_file = param_file
        self.DICOM_path = DICOM_path
        self.npz_file = npz_file
        self.pruning = False
        self.number_of_iterations = number_of_iterations
        self.iterations_update = iterations_update
        self.bio_opt = bio_opt
        self.robust_opt = robust_opt
        self.robust_eval = robust_eval
        self.alpha_beta = alpha_beta
        self.robust_bio = False
        self.robust_bio_eval = False
        self.let_co = let_co
        self.biological_model = bio_model
        self.single = single
        self.phys_dose = plot_physical_dose
        self.fractions = float(args.fractions)
        self.robust_iterations = int(robust_iterations)
        self.param_file = opt_param_file
        self.eq_weight = eq_weight
        self.min_bound = 15000
        self.max_bound = 100000000
        self.norm = norm

	#if norm:
		#print("Normalization criteria: ROI: {}, Dose: {}Gy".format(norm[0], norm[1])) # Endret litt på denne - Erlend 

        self.func_tolerance = 0
        self.gradient_tolerance = 0

        if parallell != None:
            self.num_processes = int(parallell)
            self.parallell = True
        else:
            self.parallell = False

        if continue_file != "":
            self.continue_file = continue_file
            self.continue_from_file = True
        else:
            self.continue_from_file = False

    def run(self):
        """Run function needed for the GUI"""
        self.optimize()

    def set_iterations(self, number_of_iterations, iterations_update):
        """Function that updates the number of iteration.
        Needed for the GUI"""
        self.number_of_iterations = number_of_iterations
        self.iterations_update = iterations_update

    def print_settings(self):
        print("####### Settings for the optimization #######")

        print("{:<35s} {:<s}".format("Name for optimization", self.path))
        print("{:<35s} {:<s}".format(
            "Name of objective/constraint-file", self.param_file))
        print("{:<35s} {:<s}".format("NPZ-filename", self.npz_file))
        if self.DICOM_path != "":
            print("{:<35s} {:<s}".format(
                "DICOM-files location", self.DICOM_path))
        if self.robust_opt != "":
            print("{:<35s} {:<s}".format("Robust optimization", "True"))
            print("{:<35s} {:<s}".format(
                "Robust algorithm", self.robust_opt[0]))
            if len(self.robust_opt[0]) == "minimax":
                print("{:<35s} {:<s}".format(
                    "Robust iterations for minimax", self.num_processes))

            if len(self.robust_opt) > 1:
                print("{:<35s} {:<s}".format(
                    "Stochastic weighting of main plan", self.robust_opt[1]))
            print("{:<35s} {:<s}".format(
                "Cores in use for parallell opt", self.robust_opt[0]))

        else:
            print("{:<35s} {:<s}".format("Robust optimization", "False"))
        print("{:<35s} {:<s} Gy(RBE(1.1))".format(
            "Dose cutoff for LET-calculation ", str(self.let_co)))
        print("{:<35s} {:<s}".format("RBE-model", self.biological_model))
        print("{:<35s} {:<}".format("Plot physical dose", self.phys_dose))
        print("{:<35s} {:<}".format("Plot only per iteration", self.single))
        print("{:<35s} {:<3.0f}".format("Fractions", self.fractions))
        print("{:<35s} {:<10.0f}".format(
            "Minimum weight allowed for PB", self.min_bound))
        print("{:<35s} {:<10.0f}".format(
            "Maximum weight allowed for PB", self.max_bound))
        if self.continue_from_file:
            print("{:<35s} {:<s}".format(
                "Continue from file", self.continue_file))
        else:
            print("{:<35s} {:<s}".format(
                "Using original pencil beam weights", "True"))
        print("{:<35s} {:<10.0f}".format(
            "Iteration interval for plotting", self.iterations_update))
        print("{:<35s} {:<10.0f}".format(
            "Max number of iterations", self.iterations_update))
        print("\n")

    def load(self):
        """Load all files needed for the optimization, like the datfiles,
        opt-parameter files and the npz files. Also load the dose"""

        # Start timer
        ticc = time.time()
        print("Loading files ...")
        # When we use DICOM or read the files for the first time, we open the
        # large NPZ-file. This variable is needed so we dont open it more times
        # than needed
        self.large_data_loaded = False

        # Make the directory for the optimization if it does not already exist
        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)

        # Make the directory for the optimization if it does not already exist
        isExist = os.path.exists(self.path+"/plotting_files")
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path+"/plotting_files")

        # Make the directory for the optimization if it does not already exist
        isExist = os.path.exists(self.path+"/pb_files")
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path+"/pb_files")

        # Make the directory for the optimization if it does not already exist
        isExist = os.path.exists(self.path+"/metrics")
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path+"/metrics")

        # Open parameterfile and read the names of the ROIs and # of PBs
        with open(self.param_file) as f:
            header = f.readline()
            self.datfile_list = f.readline().split(",")
        # Remove the \n from the last element
        self.datfile_list[-1] = self.datfile_list[-1][:-1]
        f.close()
        while ("" in self.datfile_list):
            self.datfile_list.remove("")

        # Read the rest of the table into a pandas dataframe, which can be used
        # freely later. Can also add whatever here
        self.opt_params = pd.read_csv(self.param_file, sep=',', skiprows=2)

        for j in range(len(self.opt_params)):
            if self.opt_params.loc[j, 'opt_type'] != 3 and self.opt_params.loc[j, 'opt_type'] != 4:
                self.opt_params.loc[j, 'p_d'] /= self.fractions

        # Define some stuff before reading the ROI-files
        self.ROI_cage_size = 0
        self.ROI_voxel_list = []
        self.ROI_bool_matrix = []

        # Opens the first dat-file to define the dimension of the entire ROI-cage
        self.ROI_dimensions = []
        g = open("ROI_datfiles/"+self.datfile_list[0], "r")
        header = g.readline()
        header = g.readline()

        # Size of the ROI-cage,
        self.ROI_cage_size = int(
            (float(header.split()[0]))*(float(header.split()[1]))*(float(header.split()[2])))
        bin_info = [(int(header.split()[0])),
                    (int(header.split()[1])), (int(header.split()[2]))]
        self.ROI_dimensions.append(bin_info)
        header = g.readline()
        min_table = [(float(header.split()[0])), (float(
            header.split()[1])), (float(header.split()[2]))]
        self.ROI_dimensions.append(min_table)
        header = g.readline()
        max_table = [(float(header.split()[0])), (float(
            header.split()[1])), (float(header.split()[2]))]
        self.ROI_dimensions.append(max_table)
        g.close()

        ######## INITIALIZE ROIS ##########
        # Reads through the ROI-files again so the voxels in the ROI can be added
        # to a list, used to define a matrix containing only the information in
        # each ROI. Also we have defined a boolean matrix which contains
        # a boolean matrix where the values are True if they are in the ROI,
        for i in range(len(self.datfile_list)):
            g = open("ROI_datfiles/"+self.datfile_list[i], "r")
            header = g.readline()
            header = g.readline()
            for j in range(2):
                header = g.readline()
            ROI_voxels = int(g.readline())
            lines = g.readlines()
            temp_vox_list = []
            line_number = np.zeros([len(lines)])

            temp_ROI_bool_matrix = np.zeros([self.ROI_cage_size])
            for i in range(len(lines)):
                splitted_lines = lines[i].split()
                temp_vox_list.append(int(splitted_lines[0])-1)
                temp_ROI_bool_matrix[int(splitted_lines[0])-1] = 1
            self.ROI_voxel_list.append(temp_vox_list)
            self.ROI_bool_matrix.append(temp_ROI_bool_matrix)
            g.close()
        ############################

        ### Read dosefile from simulation, or open existing matrixes ###

        # If the dosefile has already been read and stored in numpy-format:

        self.iterations = 1
        self.sep_npz_file = "npz_data/sep_"+self.npz_file
        item_list = ["dose", "dose_proton", "let_times_dosep"]

        self.sep_dose_list = np.empty(len(self.datfile_list), dtype=object)
        self.sep_dose_p_list = np.empty(len(self.datfile_list), dtype=object)
        # self.sep_let_pr_primary = np.empty(len(self.datfile_list),dtype = object)
        self.sep_let_times_dosep = np.empty(
            len(self.datfile_list), dtype=object)
        # First check if the original .npz file exists.
        tic = time.time()

        # If the npz-files with seperate values exist
        # print ("Reading numpy file ... {}".format(self.sep_npz_file))
        # print (len(self.datfile_list))
        # Read the data from the files
        for i in range(len(self.datfile_list)):

            self.sep_dose_list[i] = load_npz(
                "npz_data/{}_dose_{}".format(i, self.npz_file))
        for i in range(len(self.datfile_list)):

            self.sep_dose_p_list[i] = load_npz(
                "npz_data/{}_dose_proton_{}".format(i, self.npz_file))
        for i in range(len(self.datfile_list)):

            self.sep_let_times_dosep[i] = load_npz(
                "npz_data/{}_let_times_dosep_{}".format(i, self.npz_file))
            # self.sep_let_pr_primary [i] = self.sep_let_times_dosep[i] /self.sep_dose_p_list[i]
        # print(self.sep_dose_list)
        # sys.exit()
            # Define the arrays that contains all dose and LET information
            # Divides by 1.1 because that is how it is defined, and thus used
            # for comparisson with original dose plan.

        if self.continue_from_file:
            self.intensities = self.get_ints_from_file(self.continue_file)
        elif self.eq_weight:
            self.intensities = np.full(
                self.sep_dose_list[0].shape[0], 10000000)
            self.intensities = scipy.sparse.csr_matrix(
                self.intensities, dtype=np.float32)
        else:

            with np.load("npz_data/Intensities.npz") as data:
                self.intensities = scipy.sparse.csr_matrix(
                    data['ints'], dtype=np.float32)

        # print(self.intensities)
            # print (data)

        # If the npz-file with seperate values do not exist, we read the
        # large file and create the seperate_npz file
        # print ("HEYEHYEEYHEHYEHEEHY")
        if self.DICOM_path != "":

            isExist = os.path.exists(self.path+"/DICOMs")
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(self.path+"/DICOMs")
            file1 = "npz_data/dose_"+self.npz_file
            file2 = "npz_data/dosep_"+self.npz_file
            file3 = "npz_data/let_times_dosep_"+self.npz_file
            if os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3):
                print("All files for DICOM exists, continuing")
            else:
                print(
                    "One or more files do not exist. Create the files needed for DICOM")
                sys.exit()

        # print ("Data missing from the npz_data-folder\nTerminating")
        # sys.exit()

            # print("Time used for reading numpy-file: {}\n".format(time.time()-tic))

        print("Finished Loading Files, time: {:.3f} s\n".format(
            time.time()-ticc))

    def calculate_init(self):
        """Calculate the initial dose and letd distribution for the plan"""
        tic = time.time()
        print("Initializing the optimization ...\n")
        self.print_settings()

        # Calculate initial DVH/LVH and cost

        # Create DVH and LVH dictionary which is going to contain all the DVHs and LVHs
        self.dvh_dict = {}
        self.lvh_dict = {}

        # If we have chosen DICOMs, we create DICOMs for the initial plan
        if self.DICOM_path != "":
            self.create_dicom(self.intensities, "Initial")

        # create a cost list, which is going to contain the cost of all objectives
        self.cost_list = []
        for i in range(len(self.opt_params)):
            # Fille the list with additional lists, one for each objective
            self.cost_list.append([])

        # Calculate the initial DVH and LVH, and add them to the dictionary
        if self.norm != "":
            self.intensities = ensure_csr_matrix(self.intensities)
            self.intensities = self.normalize(self.intensities)
        self.calculate_DVH("Initial", self.intensities)

        self.calculate_LVH("Initial", self.intensities)

        # Calculate the initial cost of the plan, and make it the current cost.
        print("#### Optimization criteria, initial cost and metrics ####")
        self.current_cost = self.print_metrics(self.intensities, 0, 0)
        print("Initial cost: {:7.3f}".format(self.current_cost))

        # Write the pb-file for the plan.
        self.write_pb_file("pb_int_{}.res".format(self.path), self.intensities)

        self.dvh_list = []
        self.lvh_list = []
        self.lvh_list.append("Initial")
        self.dvh_list.append("Initial")
        if self.phys_dose:
            self.dvh_list.append("phys_Initial")

        # NOT IN USE, should not be used until fully implemented
        if self.pruning:
            self.prune()
            self.calculate_DVH("pruned_dvh_init", self.intensities)
            self.calculate_LVH("pruned_lvh_init", self.intensities)
            cost = self.cost_function(self.intensities)
            print("Pruned cost: {}".format(cost))
            self.lvh_list.append("pruned_lvh_init")
            self.dvh_list.append("pruned_dvh_init")

        # If we robust optimize, then we start with a robust evaluation
        self.lvh_filename = "LVH_0_{}.png".format(self.path)
        self.dvh_filename = "DVH_0_{}.png".format(self.path)
        if self.robust_opt != "":
            self.collect_all_robust_scenarios()
            self.robust_evaluate("0", self.intensities)
        elif self.robust_eval:
            self.collect_all_robust_scenarios()
            self.robust_evaluate("0", self.intensities)
        else:
            # If we dont, we plot stuff,
            print("Plotting initial dose and LETd distribtion ...")

            # Define the filenames for the initial DVH and LVH plots

            # Plot the dvhs and LVHs, with the filenames, and given ranges
            ttic = time.time()
            self.plot_dvh(self.lvh_list, self.lvh_filename,
                          [0, 8], True)  # - Erlend
            ttic = time.time()
            self.plot_dvh(self.dvh_list, self.dvh_filename,
                          [0, 2.15])   # - Erlend

            # A little bit more complicated to plot through the GUI
            if args.gui:
                # plot_dose_list = []
                # plot_volume_list = []
                filename_list = []
                # dose_list,volume_list = self.plot_dvh_gui(self.dvh_list,self.dvh_filename,[0,3.2])
                # plot_dose_list.append(dose_list)
                # plot_volume_list.append(volume_list)
                filename_list.append(self.dvh_filename)
                # dose_list,volume_list = self.plot_dvh_gui(self.lvh_list,self.lvh_filename,[0,3.2],True)
                # plot_dose_list.append(dose_list)
                # plot_volume_list.append(volume_list)
                filename_list.append(self.lvh_filename)
                filename_list.append("dummy")
                self.update_dvh.emit(filename_list, self.path)
        print("Finished calculating initial dose and LETd, time: {:4.2f}".format(
            time.time()-tic))


    def optimize(self):
        """The optimization function"""

        # Define bounds, which at the moment is between 0 and 10*10.
        # We need bounds as we cant have negative pencil beam values. This
        # is where we change things if we want a minimum and max MU value.
        bounds = []
        self.intensities = self.intensities.toarray().flatten()
        for i in range(len(self.intensities)):
            bounds.append((self.min_bound, self.max_bound))
        print(type(bounds))
        print("Starting optimization ...")
        if self.robust_opt != "":
            self.robust_optimize()
            self.write_pb_file("pb_final_{}.res".format(
                self.path), self.intensities)

        elif self.robust_bio:
            self.robust_optimize_bio()
        else:
            # ############ L-BFGS-B ##############
            # Our optimization function, which now uses L-BFGS-B to minimize the intensities of the pencil beams based on the cost function and the partial derivative of the cost function.
            # Callback function is a function that gets called after each iterations.
            # a is the output from the optimization, containing the cost and new values.

            a = minimize(self.cost_function_sep, self.intensities, bounds=bounds, jac=self.cost_function_der_sep, args=(), method='L-BFGS-B',
                         callback=self.callbackF, tol=0, options={'maxcor': 100, 'maxls': 50, 'maxiter': self.number_of_iterations, 'disp': False})  # Changed maxls to 50 - Erlend
            if a.success:
                print("Success") # - Erlend
            else:
                print(a)
                print("Could not find min") # - Erlend

            print(a.message)
            self.intensities = a.x

            # After the optimization, if we have chosen DICOMs, we create the final dicoms
            if self.norm != "":
                self.intensities = ensure_csr_matrix(self.intensities)
                self.intensities = self.normalize(self.intensities)

            if args.dicom != "":
                self.create_dicom(self.intensities, "Final")
            # Calculate the final DVH and LVH
            self.calculate_DVH("Final", self.intensities)
            self.calculate_LVH("Final", self.intensities)

            # If we plot only each iteration, we create empty lists so we only
            # use the final value
            if self.single:
                self.dvh_list = []
                self.lvh_list = []

            # Add the names to the list
            if self.phys_dose:
                self.dvh_list.append("phys_Final")
            self.dvh_list.append("Final")
            self.lvh_list.append("Final")
            self.write_pb_file("pb_final_{}.res".format(
                self.path), self.intensities)

        ###### PLOT EVERYTHING ###################
            self.lvh_filename = "LVH_Final_{}.png".format(self.path)
            self.dvh_filename = "DVH_Final_{}.png".format(self.path)
            self.plot_dvh(self.lvh_list, self.lvh_filename,
                          [0, 8], True)  # - Erlend
            self.plot_dvh(self.dvh_list, self.dvh_filename,
                          [0, 2.15])  # - Erlend
            self.plot_cost("Cost_Final.png".format(
                self.iterations), self.cost_list)

            # If we use GUI, there is something different happening.
            if args.gui:
                filename_list = []
                filename_list.append(self.dvh_filename)
                filename_list.append(self.lvh_filename)
                filename_list.append("Cost_Final.png")
                self.update_dvh.emit(filename_list, self.path)

    #### Robust evaluation ########

        if self.robust_eval:
            self.robust_evaluate("Final", self.intensities)
        elif self.robust_bio_eval:
            self.robust_bio_eval()

        print("Finished optimizing")

    def callbackF(self, a):
        """Callback function: This function gets called after every ieration
        of the optimization. """

        # If current iteration is a number that goes in the number of when
        # we want to update our plot:
        if self.norm != "":
            a = ensure_csr_matrix(a)
            a = self.normalize(a)
        if self.iterations % self.iterations_update == 0:

            # Print the metrics for the optimzzation
            self.current_cost = self.print_metrics(
                a, self.iterations, self.current_cost)
            # Create DICOM if we choose it
            if args.dicom != "":
                self.create_dicom(a, self.iterations)
            # If we oinly want to plot single, we create empty lists
            if self.single:
                self.dvh_list = []
                self.lvh_list = []
            # Calculate the DVH and LVH for current iteration

            self.calculate_DVH("{}th_iteration".format(self.iterations), a)
            self.calculate_LVH("{}th_iteration".format(self.iterations), a)

            self.dvh_list.append("{}th_iteration".format(self.iterations))

            if self.phys_dose:
                self.dvh_list.append(
                    "phys_{}th_iteration".format(self.iterations))
            self.lvh_list.append("{}th_iteration".format(self.iterations))

            self.write_pb_file("pb_{}_{}.res".format(
                self.iterations, self.path), a)

            self.lvh_filename = "LVH_{}_{}.png".format(
                self.iterations, self.path)
            self.dvh_filename = "DVH_{}_{}.png".format(
                self.iterations, self.path)

            self.plot_dvh(self.lvh_list, self.lvh_filename,
                          [0, 8], True)  # - Erlend
            self.plot_dvh(self.dvh_list, self.dvh_filename,
                          [0, 2.15])   # - Erlend
            self.plot_cost("Cost_{}.png".format(
                self.iterations), self.cost_list)
            if args.gui:
                filename_list = []
                filename_list.append(self.dvh_filename)

                filename_list.append(self.lvh_filename)
                filename_list.append("Cost_{}.png".format(self.iterations))
                self.update_dvh.emit(filename_list, self.path)
                # self.update_plot.emit(["{}/{}".format(self.path,self.dvh_filename),"{}/{}".format(self.path,self.lvh_filename)])

        else:
            self.current_cost = self.print_metrics(
                a, self.iterations, self.current_cost)
        self.iterations += 1

    # filename,ROI_voxel_list,dose_pr_primary,intensities):
    def calculate_DVH(self, filename, intensities):
        """Calculates the DVH for the intensities given, and appends
        the filename to the DVH-dictionary. Does not return anything"""

        intensities = ensure_csr_matrix(intensities)
        new_array = []
        for i in range(len(self.ROI_voxel_list)):
            tic = time.time()
            # Calculate the biological dose for ROI number i
            voxel_dose, rbe = self.biological_dose(intensities, i)
            voxel_dose = sparse_to_numpy(voxel_dose)
            # print ("Dose time: {:.5f}".format(time.time()-tic))
            # print (self.sep_dose_list)
            # Multiplies the dose with the fraction and appends it to the list
            new_array.append(voxel_dose*self.fractions)
            # print (voxel_dose*self.fractions)
            # Adds the array to the dictionary

        self.dvh_dict[filename] = new_array
        # print (self.dvh_dict[filename])
        # print(self.dvh_dict[filename][0])
        # Do the same thing for phyiscal dose if that option is chosen
        if self.phys_dose:
            new_array = []
            for i in range(len(self.ROI_voxel_list)):

                voxel_dose = (intensities@self.sep_dose_list[i])*self.fractions

                new_array.append(voxel_dose)
            self.dvh_dict["phys_{}".format(filename)] = new_array

    def calculate_LVH(self, filename, intensities):
        """Calculates the LVH for the intensities given, and appends
        the filename to the LVH-dictionary. Does not return anything.
        (LETd) = sum(let x proton_dose x intensity of pb)/(sum(proton dose x intensity of pb))"""
        new_array = []
        intensities = ensure_csr_matrix(intensities)
        for i in range(len(self.ROI_voxel_list)):
            tic = time.time()

            # Calculate the LETd for ROI number i
            up = intensities@self.sep_let_times_dosep[i]  # Above the fraction
            # print (up)
            down = intensities@self.sep_dose_p_list[i]  # Below the fraction
            # print (down)
            let_d = up/down  # Achieve the LETd
            let_d = scipy.sparse.csr_matrix(let_d)
            # print (let_d)
            # print (let_d)
            # Dose value for cutoff
            dose = intensities@self.sep_dose_list[i]
            dose = sparse_to_numpy(dose)
            let_d = sparse_to_numpy(let_d)

            # Because of we divide by zero (fix sometime), we need to convert the
            # infinity-values to zero
            # let_d = np.nan_to_num(let_d)
            # Removes ny let valye from dose region below cutoff
            # print (dose)
            # print (let_d)
            # let_d = remove_low_values(let_d,dose,float(self.let_co))
            # nonzero_mask = np.array(dose[dose.nonzero()] < float(self.let_co))[0]
            # rows = let_d.nonzero()[0][nonzero_mask]
            # cols = let_d.nonzero()[1][nonzero_mask]
            # let_d[rows, cols] = 0
            let_d[dose < float(self.let_co)] = 0
            # Add the ROI-letd to the list

            new_array.append(let_d)
            # print ("let time: {:.5f}".format(time.time()-tic))

        # Add everything to the dictionary
        self.lvh_dict[filename] = new_array

    def get_metrics(self, mf, arr, struct, prescribed_dose):
        """Function to get metrics from a DVH or LVH. Takes in the metrics file,
        dvh/LVH array, structure name and prescribed dose.
        Changed most of this function - Erlend
        """
        # Changed to decending order (e.g 2, 1.9, 1.8)
        sorted_array = sorted(arr, reverse=True)
        
        # Get the total number of voxels for the given structure (struct)
        total_vox = len(sorted_array)
        
        # Get the voxel size (volume in cc)
        voxel_size = find_voxel_volume("ROI_datfiles/"+struct)
        
        # Find the total volume
        total_volume = total_vox * voxel_size

        # Make it into a numpy array
        sorted_array = np.array(sorted_array)

        # Find the number of voxels receiving 60 Gy or more after 30 fractions
        voxel_count = np.sum(sorted_array >= 60)

        # Multiply the number of voxels with the voxel size in cc to get the V_60
        v_60 = voxel_count * voxel_size

        # Define standard metrics
        mean = np.mean(sorted_array)
        median = np.median(sorted_array)
        maximum = np.max(sorted_array)
        minimum = np.min(sorted_array)

        # Calculate D_0.03cc
        d_003cc = 0

        # First find the number of voxels that makes 0.03cm³
        amount_of_voxels = (0.03 / voxel_size) 
        # amount_of_voxels is most likely a float, thus its a whole number of voxels, pluss a fraction of one voxel
        whole_voxel = int(amount_of_voxels)
        frac_voxel = amount_of_voxels - whole_voxel

        # Loop through all voxels that sums up to be 0.03cm³
        for i in range(whole_voxel + 1):
            # If it's the fraction voxel the dose in that voxel will be multipied 
            # by the % of the volume that the fraction constitutes.
            if i == whole_voxel:
                weighting_factor = (frac_voxel / amount_of_voxels)
                dose_contribution = (sorted_array[i] * weighting_factor)
                d_003cc += dose_contribution
            else:
            # Here we use (1 / amount_of_voxels) since its one full voxel dose to account for. 
                weighting_factor = (1 / amount_of_voxels)
                dose_contribution = (sorted_array[i] * weighting_factor)
                d_003cc += dose_contribution
        # For further work: The voxels that recivies the highest dose might be in the boundary of the structure. Thus, the 'amount_of_voxels' is wrong in that scenario.
        # To fix this I propose a check function that checks how much of each voxel is inside the structure (See def check_voxel).
        # Then add that voxel and its 'inside-fraction' to a list. Then keep adding voxels and inside-fractions until
        # (sum of inside-fractions) * voxel_size = 0.03cc. Then to find the D_0.03cc metric it should be a for-loop
        # that loops through the amount of voxels that sums up to be 0.03cc and multiply that by their volume contribution and dose.
        # For example if 2/3 of a voxel is inside the structure and that voxel recieves 54 Gy,
        # the contribution from this voxel would be ( 0.03 / (2/3 * voxel_size) ) * 54 Gy.

        # Additional percentile metrics. Creates a list of the different percentile metrics.
        percent_list = np.percentile(sorted_array, [100-98, 100-95, 100-40, 100-2]) # Add more if needed. NB: if you want D60% you have to do 100-60=40.
        # Remember to also add it to the 'metrics' list below.
        # Note: Remember to change the mf.write if more percentiles are added. Search for '# Write the header of the metrics file'.

        # Changed the way to calculate the V95% and V107% - Erlend
        frac = 30 # Change this if another amount of fractions is used
        v_95_percent = ( np.sum(sorted_array >= (prescribed_dose * frac * 0.95)) / total_vox ) * 100
        v_107_percent = ( np.sum(sorted_array >= (prescribed_dose * frac * 1.07)) / total_vox ) * 100

        # Write the standard metrics. Changed it to contain all metrics in one write for better overview - Erlend
        # The metrics is as follows: Structure name, total volume (cc), mean dose, median dose,
        # minium dose, maxium dose, D_0.03cc, D98%, D95%, D40%, D2%, V_60Gy, V95%, V107%
        metrics = [struct[:-4], total_volume, mean, median, minimum, maximum, d_003cc, percent_list[0], percent_list[1], percent_list[2], percent_list[3], v_60, v_95_percent, v_107_percent]

        mf.write("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*metrics))
        
        mf.write("\n")

    def plot_dvh(self, filename, figurename, scale, let_plot=False):
        """Function for plotting DVH and LVH, as well as writing metrics.
        Taks in the filename for the file want to plot (from dictionary), figurename,
        the scale of the x-axis and if it is LET plot or not."""

        volume_list = []
        dose_list = []

        for j in range(len(filename)):
            if let_plot == True:
                # The array used for this DVH from the lvh-dictionary
                arr = self.lvh_dict[filename[j]]
                # print ("let arra")
                # print (arr)
                # Define the filenames
                dvh_filename = "{}/plotting_files/{}_{}_lvh.txt".format(
                    self.path, filename[j], self.path)
                metrics_filename = "{}/metrics/{}_{}_let_metrics.csv".format(
                    self.path, filename[j], self.path)
                # Defined the resolution of the DVH.
                plotting_range = 800
            else:
                # The array used for this DVH from the lvh-dictionary
                arr = self.dvh_dict[filename[j]]
                # Define the filenames
                # print ("dose arra")
                # print (arr)
                dvh_filename = "{}/plotting_files/{}_{}_dvh.txt".format(
                    self.path, filename[j], self.path)
                metrics_filename = "{}/metrics/{}_{}_dose_metrics.csv".format(
                    self.path, filename[j], self.path)
                # Defined the resolution of the DVH.
                plotting_range = 400
            # print (arr[0])
            # Define some other stuff
            number_of_rois = len(arr)
            volume = []
            dose = []
            v_list = []
            d_list = []
            # print (arr)
            # Open the the writable files
            f = open(dvh_filename, "w")
            mf = open(metrics_filename, "w")
            # Write the header of the metrics file
            mf.write("Structure,Volume (cc),Mean dose (Gy),Median dose (Gy),Min dose (Gy),Max dose (Gy),D_0.03cc,D98% (Gy),D95% (Gy),D40% (Gy),D2% (Gy),V_60Gy (cc),V95%,V107%\n")
            # - Erlend

            # For each ROI
            for j in range(number_of_rois):
                f.write("\n")
                # Header of each ROI
                f.write("Structure: {}\n".format(self.datfile_list[j]))

                f.write("Dose [Gy]\t\tVolume [%]\n")
                # print ("wolololol")

                # Write metrics to the metrics file

                a = [float(i) for i in arr[j]]

                # Write metrics to the metrics file
                self.get_metrics(mf, a, self.datfile_list[j], 1.8) # Endret til 1.8 fra 2 - Erlend
                volume = []
                dose = []

                # print(a.shape)
                # a=arr.toarray()[0]

                volume = []
                dose = []
                # For every value in the plotting range
                for i in range(plotting_range):
                    if not let_plot:
                        # Appends the volume, times the fraction. Stepsize is 0.01 Gy per fraction
                        dose.append(float(i*0.01*self.fractions))

                        dose_level = i*0.01*self.fractions
                        # Check how much of the volume are above this dose level
                        volume.append(
                            100 * (np.sum(np.fromiter((k > dose_level for k in a), dtype=bool))) / float(len(a)))
                    else:
                        # Same here for the LETd, only without multiplication of the
                        # number of fractions
                        dose.append(float(i*0.01))

                        dose_level = i*0.01
                        volume.append(
                            100 * (np.sum(np.fromiter((k > dose_level for k in a), dtype=bool))) / float(len(a)))

                    # Write down the volume
                    f.write("{}\t\t{}\n".format(float(i*0.01), volume[-1]))
                # Add the values to list ment for plitting
                v_list.append(volume)
                d_list.append(dose)
            # Another list meant for plotting
            volume_list.append(v_list)
            dose_list.append(d_list)

        # Close files
        f.close()
        mf.close()

        # Define colors from what I asked chatGPT about the most distinctive
        # colors
#        color_list = ['black',
#                      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#                      '#8c564b', '#e377c2', '#aec7e8', '#bcbd22', '#c49c94',
#                      '#17becf', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
#                      '#dbdb8d', '#f7b6d2', '#c7c7c7', '#7f7f7f', '#9edae5'
#                      ]

        model_names_bio = []
        fig, ax = plt.subplots()
        color_list = ['#FF5959', '#1f77b4', '#2ca02c', '#9467bd', '#e377c2', '#bcbd22', '#654B7D', '#17becf', '#598DFF', '#dbdb8d', '#f7b6d2', '#c7c7c7', '#7f7f7f', '#9edae5'] # Just used this to make my plot identical to that of my own script plots

        # For every file and evey ROI, plot the DVH or LVH

        plotted_structures = []
        last = 0                                          #
        for i in range(len(filename)):
            last = i
        for i in range(len(filename)):                    #
            num_structures = len(dose_list[i])            #
            for j in range(num_structures):               #

                # Sjekker at ikke siste char i navnet er en digit
                base_name = self.datfile_list[j][:-4]
                if base_name[-1].isdigit():               # - Erlend
                    new_base_name = base_name[:-1]        #
                else:                                     #
                    new_base_name = base_name             #
                if i == 0:
                    a = ax.plot(dose_list[i][j], volume_list[i][j], color=color_list[j],
                                linestyle="--", label=new_base_name, linewidth=6, alpha=1.0)
                elif i == last:
                    a = ax.plot(dose_list[i][j], volume_list[i][j], color=color_list[j],
                                linestyle="-", label=None, linewidth=6, alpha=1.0)

                else:
                    a = ax.plot(dose_list[i][j], volume_list[i][j], color=color_list[j],
                                linestyle=":", label=None, linewidth=4, alpha=1.0)
                # Endret slik at det ikke kommer "1" og "-final" i labels. For eksempel "brainstem1 - final" blir "brainstem" - Erlend

        # Plot properties
        ax.set_ylabel("Volume [%]", size=txtsize)
        if let_plot == True:
            # Fjernet '' - Erlend
            ax.set_xlabel("LET$_d$[keV/$\mu$m]", size=txtsize)
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        else:
            # Endret fra RBE-weighted dose til Dose - Erlend
            ax.set_xlabel("Dose [Gy(RBE)]", size=txtsize)
            ax.xaxis.set_minor_locator(MultipleLocator(0.1*self.fractions))
        ax.grid(color='black', alpha=0.5)
        if let_plot:
            ax.set_xlim(scale)
        else:
            ax.set_xlim([scale[0]*self.fractions, scale[1]*self.fractions])
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
        ax.legend(loc='upper right', fancybox=True,
                  shadow=True, ncol=1, fontsize=lgndsize)

        fig = plt.gcf()
        left = 0.09
        right = 0.95
        bottom = 0.18
        top = 0.93
        wspace = 0.15
        hspace = 0.24
        fig.subplots_adjust(left=left, right=right, bottom=bottom,
                            top=top, wspace=wspace, hspace=hspace)
        fig.set_size_inches((28, 17))

        fig.savefig("{}/{}".format(self.path, figurename))
        plt.close('all')

    def plot_cost(self, figurename, cost_list, robust=False):
        """Plot the cost function, both for robust optimization and regular
        A list of cost for each iteration is saved throughout the optimization,
        which is here plotted. For robust optimization, the costs for the
        differen uncertainties is plotted, while for regular optimization,
        the cost for the different objectives are plotted"""
        fig, ax = plt.subplots()

        color_list = ["black", "blue", "red", "darkgreen", "orangered",
                      "darkgoldenrod", "khaki", "darkkhaki", "gold",
                      "yellowgreen", "greenyellow", "lawngreen", "green",
                      "lightblue", "deepskyblue", "dodgerblue", "blue",
                      "violet", "fuchsia", "deeppink", "purple",
                      "black", "gray", "silver", "gainsboro", "green", "yellow", "c", "g", "purple", "orange", "yellow", "teal", "black", "black", "black", "black", "black", "black"] # Added fill colors in the end since my number of objectives was to long for the color list. This part should be changed to a colormap so it doesn't crash when more objectives are added. - Erlend
        x_axis = []
        if not cost_list:
            return
        plot_limit = 20
        if robust:
            # Add values for x axis, equal to length of cost list
            for i in range(len(cost_list)):
                x_axis.append(i)

            # Arrange the costs so it can be plotted later
            plot_list = []
            # All the different uncertainties + sum:
            for i in range(len(cost_list[0])):
                temp_plot_list = []
                for j in range(len(cost_list)):  # All iterations
                    temp_plot_list.append(cost_list[j][i])
                plot_list.append(temp_plot_list)

            # Plot the cost for every uncertainty +1 (the sum of costs)

            for i in range(len(plot_list)+1):
                # If this is the last element+1, we plot the total cost on
                # a seperate axis
                if i == len(plot_list):
                    ax2 = ax.twinx()
                    if len(sum_list) > plot_limit:
                        ax2.plot(x_axis[-plot_limit:], sum_list[-plot_limit:],
                                 color="black", linestyle="--", label="Total cost", linewidth=lw)
                    else:
                        ax2.plot(x_axis, sum_list, color="black",
                                 linestyle="--", label="Total cost", linewidth=lw)
                # The first cost is always the original plan, so plot that first
                elif i == 0:
                    if len(plot_list[i]) > plot_limit:
                        ax.plot(x_axis[-plot_limit:], plot_list[i][-plot_limit:],
                                color=color_list[i], linestyle="-", label="Original_plan", linewidth=lw)
                    else:
                        ax.plot(x_axis, plot_list[i], color=color_list[i],
                                linestyle="-", label="Original_plan", linewidth=lw)

                    sum_list = np.array(plot_list[i])
                else:

                    if len(plot_list[i]) > plot_limit:
                        ax.plot(x_axis[-plot_limit:], plot_list[i][-plot_limit:], color=color_list[i],
                                linestyle="-", label="Uncertainty {}".format(i), linewidth=lw)
                    else:
                        ax.plot(x_axis, plot_list[i], color=color_list[i],
                                linestyle="-", label="Uncertainty {}".format(i), linewidth=lw)
                    sum_list = sum_list + np.array(plot_list[i])
            ax2.set_ylabel("Total Cost", size=txtsize)
            ax2.tick_params(axis='both', which='both', labelsize=ticksize)
            ax2.legend(loc='upper center', fancybox=True,
                       shadow=True, ncol=1, fontsize=lgndsize)
        else:
            # If not robust, just plot regular
            for i in range(len(cost_list[0])):
                x_axis.append(i)

            for i in range(len(cost_list)):
                if len(cost_list[i]) > plot_limit:
                    if i == 0:
                        ax.plot(x_axis[-plot_limit:], cost_list[i][-plot_limit:],
                                color=color_list[i], linestyle="-", label="Total cost", linewidth=lw)
                    else:
                        ax.plot(x_axis[-plot_limit:], cost_list[i][-plot_limit:], color=color_list[i],
                                linestyle="-", label="Objective {}".format(i), linewidth=lw)
                else:
                    if i == 0:
                        ax.plot(x_axis, cost_list[i], color=color_list[i],
                                linestyle="-", label="Total cost", linewidth=lw)
                    else:
                        ax.plot(x_axis, cost_list[i], color=color_list[i],
                                linestyle="-", label="Objective {}".format(i), linewidth=lw)

        # Plot properties here
        ax.set_ylabel("Individual Cost", size=txtsize)
        ax.set_xlabel("Iteration", size=txtsize)
        ax.grid(color='black', alpha=0.5)
        if len(x_axis) > plot_limit:

            ax.set_xlim([len(x_axis)-plot_limit, len(x_axis)])
        else:
            ax.set_xlim([0, len(x_axis)+1])
        ax.tick_params(axis='both', which='both', labelsize=ticksize)

        ax.legend(loc='upper right', fancybox=True,
                  shadow=True, ncol=1, fontsize=lgndsize)

        fig = plt.gcf()
        left = 0.09
        right = 0.95
        bottom = 0.18
        top = 0.93
        wspace = 0.15
        hspace = 0.24
        fig.subplots_adjust(left=left, right=right, bottom=bottom,
                            top=top, wspace=wspace, hspace=hspace)
        fig.set_size_inches((28, 17))

        fig.savefig("{}/{}".format(self.path, figurename))
        # Close all figures or the progrram becomes mad
        plt.close('all')

    # filename,figurename,scale):
    def plot_dvh_gui(self, filename, figurename, scale, let_plot=False):
        """Prepare arrays for the GUI to plot. Due to some restrictions in
        matplotlib, we cant plot thing outside the main thread. This function is
        similar to the normal plot dvh-function"""
        volume_list = []
        dose_list = []
        for j in range(len(filename)):
            if let_plot == True:
                arr = self.lvh_dict[filename[j]]
                dvh_filename = "{}/{}_{}_lvh.txt".format(
                    self.path, filename[j], self.path)
            else:
                arr = self.dvh_dict[filename[j]]
                dvh_filename = "{}/{}_{}_dvh.txt".format(
                    self.path, filename[j], self.path)

            number_of_rois = len(arr)
            volume = []
            dose = []
            v_list = []
            d_list = []

            f = open(dvh_filename, "w")
            for j in range(number_of_rois):
                f.write("\n")
                f.write("Structure: {}\n".format(self.datfile_list[j]))
                f.write("Dose [Gy]\t\tVolume [%]\n")
                a = [float(i) for i in arr[j]]

                volume = []
                dose = []
                for i in range(600):
                    dose.append(float(i*0.01))

                    dose_level = i*0.01
                    volume.append(
                        100 * (np.sum(np.fromiter((k > dose_level for k in a), dtype=bool))) / float(len(a)))
                    f.write("{}\t\t{}\n".format(float(i*0.01), volume[-1]))
                v_list.append(volume)
                d_list.append(dose)

            volume_list.append(v_list)
            dose_list.append(d_list)

        f.close()
        return dose_list, volume_list

    def plot_dvh_ro(self, plot_dose_list, plot_volume_list, figurename):
        """Plot DVH for robust optimization"""

        for k in range(2):
            fig, ax = plt.subplots()
            for i in range(len(plot_dose_list)):  # Number of robust scenarios
                if plot_dose_list[i] != None:
                    # Dose and let list for each robust scenario
                    dose_list = plot_dose_list[i][k]
                    volume_list = plot_volume_list[i][k]

                    color_list = ['black',
                                  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                  '#8c564b', '#e377c2', '#aec7e8', '#bcbd22', '#c49c94',
                                  '#17becf', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                                  '#dbdb8d', '#f7b6d2', '#c7c7c7', '#7f7f7f', '#9edae5'
                                  ]  # Endret til samme som den uten robust bruker - Erlend

                    # Plot all ROIs for robust plan.

                    for j in range(len(dose_list[0])):
                        # Sjekker at ikke siste char i navnet er en digit
                        base_name = self.datfile_list[j][:-4]
                        if base_name[-1].isdigit():            # - Erlend
                            new_base_name = base_name[:-1]     #
                        else:                                  #
                            new_base_name = base_name          #

                            # If first plan (main plan), use stronger color (higher alf)
                        if i == 0:
                            a = ax.plot(dose_list[-1][j], volume_list[-1][j], color=color_list[j],
                                        linestyle="-", label=new_base_name, linewidth=5, alpha=1.0)
                        else:
                            a = ax.plot(dose_list[-1][j], volume_list[-1][j], color=color_list[j],
                                        linestyle="-", label=None, linewidth=3, alpha=0.4)
                        # - Erlend

            # Plot properties
            case = ["a)", "b)"]
            ROI_list = ["let pr beam", "let mean",
                        "dose-averaged LET per pb per vox", "dose per pb per vox"]
            ax.set_ylabel("Volume [%]", size=txtsize)
            if k == 1:
                # Fjernet '' - Erlend
                ax.set_xlabel("LET$_d$[keV/$\mu$m]", size=txtsize)
                name_of_file = "{}/{}.png".format(self.path, figurename[k])
                ax.set_xlim([0, 8])  # Endret til 8 fra 6 - Erlend
                ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            else:
                ax.set_xlabel("Dose [Gy(RBE)]", size=txtsize)
                name_of_file = "{}/{}.png".format(self.path, figurename[k])
                # ax.set_xlim([0,3*self.fractions]) - Erlend
                ax.set_xlim([0, 70])  # Endret til 70 - Erlend
                ax.xaxis.set_minor_locator(MultipleLocator(0.1*self.fractions))
            ax.grid(color='black', alpha=0.5)

            ax.tick_params(axis='both', which='major', labelsize=ticksize)
            ax.legend(loc='upper right', fancybox=True,
                      shadow=True, ncol=1, fontsize=lgndsize)

            fig = plt.gcf()
            left = 0.09
            right = 0.95
            bottom = 0.18
            top = 0.93
            wspace = 0.15
            hspace = 0.24
            fig.subplots_adjust(
                left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
            fig.set_size_inches((28, 17))
            fig.savefig(name_of_file)

    def collect_all_robust_scenarios(self):
        """NB CHANGE NAME OF FUNCTION THAT GETS CALLES"""
        tic = time.time()
        # As of now, this is how all robust npz-files are defined
        ro_file_list_all = [self.npz_file,
                            self.npz_file[:-4]+"_pluss{}.npz".format("_0,0,0"),
                            self.npz_file[:-4] +
                            "_pluss{}.npz".format("_0,0.3,0"),
                            self.npz_file[:-4] +
                            "_pluss{}.npz".format("_0,-0.3,0"),
                            self.npz_file[:-4] +
                            "_pluss{}.npz".format("_0,0,0.3"),
                            self.npz_file[:-4] +
                            "_pluss{}.npz".format("_0,0,-0.3"),
                            self.npz_file[:-4] +
                            "_pluss{}.npz".format("_0.3,0,0"),
                            self.npz_file[:-4] +
                            "_pluss{}.npz".format("_-0.3,0,0"),
                            self.npz_file[:-4]+"{}.npz".format("_0,0,0"),
                            self.npz_file[:-4]+"{}.npz".format("_0,0.3,0"),
                            self.npz_file[:-4]+"{}.npz".format("_0,-0.3,0"),
                            self.npz_file[:-4]+"{}.npz".format("_0,0,0.3"),
                            self.npz_file[:-4]+"{}.npz".format("_0,0,-0.3"),
                            self.npz_file[:-4]+"{}.npz".format("_0.3,0,0"),
                            self.npz_file[:-4]+"{}.npz".format("_-0.3,0,0"),
                            self.npz_file[:-4]+"_minus{}.npz".format("_0,0,0"),
                            self.npz_file[:-4] +
                            "_minus{}.npz".format("_0,0.3,0"),
                            self.npz_file[:-4] +
                            "_minus{}.npz".format("_0,-0.3,0"),
                            self.npz_file[:-4] +
                            "_minus{}.npz".format("_0,0,0.3"),
                            self.npz_file[:-4] +
                            "_minus{}.npz".format("_0,0,-0.3"),
                            self.npz_file[:-4] +
                            "_minus{}.npz".format("_0.3,0,0"),
                            self.npz_file[:-4]+"_minus{}.npz".format("_-0.3,0,0")]

        # Create a new list to store the existing filenames
        self.ro_file_list = []
        self.cost_list = []
        ro_file_param_list = []
        # Iterate through the file list and check if each file exists
        for filename in ro_file_list_all:
            if os.path.exists("npz_data/dose_"+filename):
                # Add files that exist to list.
                self.ro_file_list.append(filename)

    def robust_evaluate(self, name, intensities):
        """Function that evaluates a plans robstness.
        Is similar to the robust optimization function, only
        without the optimization part"""

        intensities = ensure_csr_matrix(intensities)
        if self.norm != "":
            intensities = self.normalize(intensities)

        if args.dicom != "":
            self.create_dicom(intensities, name)
        print("Robust evaluating ...")
        tic = time.time()
        arguments = zip(self.ro_file_list,
                        itertools.repeat(intensities),
                        itertools.repeat(self.let_co),
                        itertools.repeat(self.ROI_voxel_list),
                        itertools.repeat(self.opt_params),
                        itertools.repeat(self.ROI_cage_size),
                        itertools.repeat(self.path),
                        itertools.repeat(self.datfile_list),
                        itertools.repeat(name),
                        itertools.repeat(self.alpha_beta),
                        itertools.repeat(self.biological_model),
                        itertools.repeat(self.fractions),
                        itertools.repeat(self.robust_opt))
        dose_list = []
        volume_list = []
        print("Creating a pool of {} processess for parallell evaluation".format(
            self.num_processes))
        pool = multiprocessing.Pool(processes=self.num_processes)
        results = list(pool.imap(evaluate_robust, arguments))
        pool.close()
        pool.join()
        for i in range(len(results)):
            dose_list.append(results[i][0])
            volume_list.append(results[i][1])

        cost = []
        pool = multiprocessing.Pool(processes=self.num_processes)
        cost = pool.map(coll_cost_function, arguments)
        pool.close()
        pool.join()
        self.plot_dvh_ro(dose_list, volume_list, ["DVH_{}_{}".format(
            name, self.path), "LVH_{}_{}".format(name, self.path)])
        self.plot_cost("Cost_{}.png".format(name), self.cost_list, True)
        if args.gui:
            self.update_dvh.emit(["DVH_{}_{}.png".format(name, self.path), "LVH_{}_{}.png".format(
                name, self.path), "Cost_{}.png".format(name)], self.path)

        print("Finished robust evaluating, time: {}".format(time.time()-tic))

    def robust_optimize(self, evaluation=False):
        """Function for robust optimization"""
        self.cost_list = []
        self.func_tolerance = 0.1
        self.converged = False
        print(self.robust_opt)
        if len(self.robust_opt) > 1:
            self.stochastic_weighting = float(self.robust_opt[1])
        else:
            self.stochastic_weighting = 1
        if self.robust_opt[0] == "stochastic":
            self.stochastic_robust_optimization()
        elif self.robust_opt[0] == "minimax":
            self.minimax_robust_optimization()
        else:
            print("Not a valid robust optimization algorithm.")
            print("Choose stochastic og minimax")
            sys.exit()

    def minimax_robust_optimization(self):

        # Create minimum and maximum values for intensities.
        name = "Dummy"
        bounds = []
        for i in range(len(self.intensities)):
            bounds.append((self.min_bound, self.max_bound))
        self.collected_dose_list = [self.sep_dose_list, "dummy"]
        self.collected_dose_p_list = [self.sep_dose_p_list, "dummy"]
        self.collected_let_times_dosep_list = [
            self.sep_let_times_dosep, "dummy"]

        for j in range(self.number_of_iterations):
            iteration_time = time.time()

            # Define that we should evaluate the first iteration.
            self.iterations = 1
            if j % self.iterations_update == 0:
                if j != 0:
                    evaluate = True
                else:
                    evaluate = False
            else:
                evaluate = False

            wpi = 0  # worst_plan_index
            robust_cost = 0
            total_cost = 0

            arguments = zip(self.ro_file_list,
                            itertools.repeat(self.intensities),
                            itertools.repeat(self.let_co),
                            itertools.repeat(self.ROI_voxel_list),
                            itertools.repeat(self.opt_params),
                            itertools.repeat(self.ROI_cage_size),
                            itertools.repeat(self.path),
                            itertools.repeat(self.datfile_list),
                            itertools.repeat(name),
                            itertools.repeat(self.alpha_beta),
                            itertools.repeat(self.biological_model),
                            itertools.repeat(self.fractions),
                            itertools.repeat(self.robust_opt))

            print("Creating a pool of {} processess for parallell evaluation".format(
                self.num_processes))
            pool = multiprocessing.Pool(processes=self.num_processes)
            # Use the Pool to parallelize reading the npz files, with these parameters

            costs = pool.map(coll_cost_function, arguments)
            pool.close()
            pool.join()

            print("Total reading time: {}".format(time.time()-iteration_time))

            worst_cost = 0
            # Determine which plan has the worst cost
            for i in range(len(costs)):
                if costs[i] != None:
                    print("Cost plan {:2}: {:.4f}".format(i, costs[i]))
                    if costs[i] > worst_cost:
                        worst_cost = costs[i]
                        wpi = i
            # Add all costs to the cost list
            self.cost_list.append(costs)

            if j == 0:
                prev_cost = np.sum(costs)
            else:
                prev_cost = total_sum
            total_sum = np.sum(costs)
            difference = 100*((prev_cost-total_sum)/prev_cost)
            if difference < 0 and j > 0:
                print("WARNING: DIVERGING COST FUNCTION, CONSIDER EXITING")
            elif difference < self.func_tolerance and j > 0:
                print("CONVERGED")
                break

            # Checking if we want to evaluate/plot the current iteration
            if evaluate:
                print("EVALUATING AND UPDATING PLOT . . .")
                self.robust_evaluate(
                    "{}th_iteration".format(j), self.intensities)
                if self.DICOM_path != "":
                    self.create_dicom(self.intensities,
                                      "{}th_iteration".format(j))

            # Print information
            print("Using file {}, with cost {:.5}, and a total cost for all plans: {:.5}\n".format(
                self.ro_file_list[wpi].split("/")[-1], costs[wpi], total_sum))
            print("Time used: {}".format(time.time()-iteration_time))
            print("Starting optimization for robust iteration {}:\n".format(j))
            # Read the file which provided the highest cost.

            self.wpi = wpi

            sep_dose_list = np.empty(len(self.datfile_list), dtype=object)
            sep_dose_p_list = np.empty(len(self.datfile_list), dtype=object)
            sep_let_times_dosep = np.empty(
                len(self.datfile_list), dtype=object)
            # First check if the original .npz file exists.
            for i in range(len(self.datfile_list)):
                sep_dose_list[i] = load_npz(
                    "npz_data/{}_dose_{}".format(i, self.ro_file_list[wpi]))
                sep_dose_p_list[i] = load_npz(
                    "npz_data/{}_dose_proton_{}".format(i, self.ro_file_list[wpi]))
                sep_let_times_dosep[i] = load_npz(
                    "npz_data/{}_let_times_dosep_{}".format(i, self.ro_file_list[wpi]))

            self.collected_dose_list[1] = sep_dose_list
            self.collected_dose_p_list[1] = sep_dose_p_list
            self.collected_let_times_dosep_list[1] = sep_let_times_dosep

            # Minimize the plan with highest cost, the number of iterations depends on the chosen value
            a = minimize(self.cost_function_ro, self.intensities, bounds=bounds, jac=self.cost_function_der_ro, args=(), method='L-BFGS-B',
                         callback=self.callbackF_ro, options={'maxcor': 100, 'maxls': 50, 'eps': 10000, "ftol": 0, "gtol": 0, 'maxiter': self.robust_iterations, 'disp': False})  # Endret på maxls fra 15 til 50 - Erlend

            self.intensities = a.x

            print("Time for iteration {}: {}".format(
                j, time.time()-iteration_time))

        # After the optimization, evaluate the plans.
        # Similar to above, only we are not optimizing the code
        print("Creating a pool of {} processess for parallell evaluation".format(
            self.num_processes))
        pool = multiprocessing.Pool(processes=self.num_processes)

        # Use the Pool to parallelize reading the npz files

        self.robust_evaluate("Final", self.intensities)
        if self.DICOM_path != "":
            self.create_dicom(self.intensities, "Final")

    def stochastic_robust_optimization(self):
        """Something here"""
        bounds = []
        for i in range(len(self.intensities)):
            bounds.append((self.min_bound, self.max_bound))

        a = minimize(self.collective_cost_function, self.intensities, bounds=bounds, jac=self.collective_cost_function_der, args=(), method='L-BFGS-B',
                     callback=self.callbackF_sto_ro, options={'maxcor': 100, 'maxls': 50, 'eps': 10000, "ftol": 0, "gtol": 0, 'maxiter': self.number_of_iterations, 'disp': False})  # Endret på maxls fra 15 til 50 - Erlend
        print(a.message)
        self.intensities = a.x
        self.robust_evaluate("Final", a.x)
        if self.DICOM_path != "":
            self.create_dicom(self.intensities, "Final")

    def collective_cost_function(self, x):
        """Function to collect cost for a list of doses, based on different scenarios"""
        collected_cost = []
        self.robust_let = True
        name = "dummy_variable"
        if self.converged:
            return self.closing_cost
        arguments = zip(self.ro_file_list,
                        itertools.repeat(x),
                        itertools.repeat(self.let_co),
                        itertools.repeat(self.ROI_voxel_list),
                        itertools.repeat(self.opt_params),
                        itertools.repeat(self.ROI_cage_size),
                        itertools.repeat(self.path),
                        itertools.repeat(self.datfile_list),
                        itertools.repeat(name),
                        itertools.repeat(self.alpha_beta),
                        itertools.repeat(self.biological_model),
                        itertools.repeat(self.fractions),
                        itertools.repeat(self.robust_opt))

        pool = multiprocessing.Pool(processes=self.num_processes)
        collected_cost = pool.map(coll_cost_function, arguments)
        pool.close()
        pool.join()
        self.stoch_cost = collected_cost
        return np.sum(collected_cost)

    def collective_cost_function_der(self, x):
        """Function to collect the derivative of the cost for a list of doses, based on different scenarios"""
        name = "dummy_variable"
        self.robust_let = True
        arguments = zip(self.ro_file_list,
                        itertools.repeat(x),
                        itertools.repeat(self.let_co),
                        itertools.repeat(self.ROI_voxel_list),
                        itertools.repeat(self.opt_params),
                        itertools.repeat(self.ROI_cage_size),
                        itertools.repeat(self.path),
                        itertools.repeat(self.datfile_list),
                        itertools.repeat(name),
                        itertools.repeat(self.alpha_beta),
                        itertools.repeat(self.biological_model),
                        itertools.repeat(self.fractions),
                        itertools.repeat(self.robust_opt))

        if self.converged:
            return np.zeros(x.size)
        pool = multiprocessing.Pool(processes=self.num_processes)
        results = pool.map(coll_cost_function_der, arguments)

        pool.close()
        pool.join()
        temp_results = np.sum(results, axis=0)
        return temp_results

    def callbackF_ro(self, a):
        """Callback funtion for the robus toptimzation. A little less complicated"""
        self.current_cost = self.print_metrics(
            a, self.iterations, self.current_cost, False)
        self.iterations += 1

    def callbackF_sto_ro(self, a):
        """Callback funtion for the robus toptimzation. A little less complicated"""
        if self.converged != True:
            self.func_tolerance = 0.1  # %
            if self.iterations == 1:
                prev_cost_list = self.stoch_cost
            else:
                prev_cost_list = self.new_stoch_cost

            for i in range(len(self.ro_file_list)):
                prev_cost = prev_cost_list[i]
                cost = self.stoch_cost[i]

                # print (prev_cost,cost)
                print("Cost for {} after {:.0f} iterations {:.2f}	{:.2f}% difference in cost".format(
                    self.ro_file_list[i], self.iterations, cost, 100*(prev_cost-cost)/prev_cost))
            self.cost_list.append(self.stoch_cost)
            prev_cost = np.sum(prev_cost_list)
            cost = np.sum(self.stoch_cost)
            difference = 100*((prev_cost-cost)/prev_cost)
            if difference < self.func_tolerance and self.iterations > 1:

                self.converged = True
                self.closing_cost = np.sum(self.new_stoch_cost)
            print("Total cost after {0:.0f} iterations for all plans: {1:.2f}	{2:.2f}% difference in cost\n\n".format(
                self.iterations, cost, 100*(prev_cost-cost)/prev_cost))
            if self.iterations % self.iterations_update == 0:
                print("Evaluating iteration number {}\n".format(self.iterations))
                self.robust_evaluate(
                    "{}th_iteration".format(self.iterations), a)
                self.write_pb_file("pb_{}_{}.res".format(
                    self.iterations, self.path), a)
                if self.DICOM_path != "":
                    self.create_dicom(self.intensities,
                                      "{}th_iteration".format(self.iterations))
            self.new_stoch_cost = self.stoch_cost
            self.iterations += 1
        else:
            print("Function converged: Difference in cost between iterations < {}%".format(
                self.func_tolerance))

    def read_npz_file(self, name):
        """Function to read a npz-file, but not the intensity"""
        tic = time.time()
        data = np.load(name)

        self.dose_pr_primary = data['ds_pr_pr']
        self.dose_proton_pr_primary = data['dsp_pr_pr']
        self.let_pr_primary = data['let_values']
        self.let_times_dosep = self.let_pr_primary * self.dose_proton_pr_primary

        print("Time used for reading numpy-file: {}".format(time.time()-tic))

    def read_sep_npz_file(self, name):
        """Function to read a sep_npz-file, but not the intensity"""

        print("Reading: {}".format(name))
        for i in range(len(self.datfile_list)):

            self.sep_dose_list[i] = load_npz(
                "npz_data/{}_dose_{}".format(i, name))
        for i in range(len(self.datfile_list)):

            self.sep_dose_p_list[i] = load_npz(
                "npz_data/{}_dose_proton_{}".format(i, name))
        for i in range(len(self.datfile_list)):

            self.sep_let_times_dosep[i] = load_npz(
                "npz_data/{}_let_times_dosep_{}".format(i, name))

        print("Time used for reading numpy-file: {}".format(time.time()-tic))

    def robust_optimize_bio(self):
        """Robust biological optimization, not finished"""
        ro_iterations = 3
        bio_model_list = args.robust_bio
        print(bio_model_list)
        ab_list = [10]
        for j in range(ro_iterations):
            wp_mi = 0  # worst_plan_model_index
            wp_abi = 0  # worst_plan_ALPHA_BETAindex
            robust_cost = 0
            total_cost = 0
            for i in range(len(bio_model_list)):
                self.biological_model = bio_model_list[i]
                for k in range(len(ab_list)):
                    self.alpha_beta = float(ab_list[k])
                    if j == 0:
                        self.calculate_DVH("int_ro_{}_{}".format(
                            bio_model_list[i], self.alpha_beta), self.intensities)
                        self.calculate_LVH("int_ro_{}_{}".format(
                            bio_model_list[i], self.alpha_beta), self.intensities)
                        self.dvh_list.append("int_ro_{}_{}".format(
                            bio_model_list[i], self.alpha_beta))
                        self.lvh_list.append("int_ro_{}_{}".format(
                            bio_model_list[i], self.alpha_beta))
                    cost = self.cost_function(self.intensities)
                    total_cost += cost
                    print("Cost: {} for plan {} with a/b of {}".format(cost,
                          bio_model_list[i], self.alpha_beta))
                    if robust_cost < cost:

                        robust_cost = cost
                        wp_mi = i
                        wp_abi = k
            print("Using the model {} and a/b of {}, with cost {}, and a total cost for all plans: {}".format(
                bio_model_list[wp_mi], ab_list[wp_abi], robust_cost, total_cost))
            self.biological_model = bio_model_list[wp_mi]
            self.alpha_beta = float(ab_list[wp_abi])
            a = minimize(self.cost_function, self.intensities, jac=self.cost_function_der, args=(), method='L-BFGS-B', options={
                         'maxcor': 1, 'maxls': 50, 'eps': 1e-4, "ftol": self.func_tolerance, "gtol": self.gradient_tolerance, 'maxiter': 3, 'disp': True})  # Endret på maxls fra 15 til 50 - Erlend
            self.intensities = a.x

        for i in range(len(bio_model_list)):
            self.biological_model = bio_model_list[i]
            for k in range(len(ab_list)):
                self.alpha_beta = float(ab_list[k])
                self.calculate_DVH("lfbgs_ro_bio_eval_{}_{}".format(
                    bio_model_list[i], self.alpha_beta), self.intensities)
                self.calculate_LVH("lfbgs_ro_bio_eval_{}_{}".format(
                    bio_model_list[i], self.alpha_beta), self.intensities)
                self.dvh_list.append("lfbgs_ro_bio_eval_{}_{}".format(
                    bio_model_list[i], self.alpha_beta))
                self.lvh_list.append("lfbgs_ro_bio_eval_{}_{}".format(
                    bio_model_list[i], self.alpha_beta))
        self.write_pb_file("pb_final_lfbgs_ro_bio_eval_{}_{}.res".format(
            bio_model_list[i], self.alpha_beta), self.intensities)

    def robust_bio_eval(self):
        """Robust biological evaluation, not finished"""
        ro_iterations = 3
        bio_model_list = args.robust_bio
        print(bio_model_list)
        ab_list = [10]
        for i in range(len(bio_model_list)):
            self.biological_model = bio_model_list[i]
            for k in range(len(ab_list)):
                self.alpha_beta = float(ab_list[k])
                self.calculate_DVH("lfbgs_ro_bio_{}_{}".format(
                    bio_model_list[i], self.alpha_beta), self.intensities)
                self.calculate_LVH("lfbgs_ro_bio_{}_{}".format(
                    bio_model_list[i], self.alpha_beta), self.intensities)
                self.dvh_list.append("lfbgs_ro_bio_{}_{}".format(
                    bio_model_list[i], self.alpha_beta))
                self.lvh_list.append("lfbgs_ro_bio_{}_{}".format(
                    bio_model_list[i], self.alpha_beta))

    def cost_function_sep(self, x):
        """Cost function that we try to minimize.
        The input x is the array of current intensities"""
        tic = time.time()
        cost = 0

        x = scipy.sparse.csr_matrix(x)
        # For every objective/constraint
        for j in range(len(self.opt_params)):
            # i is the number of the ROI
            i = self.opt_params.loc[j, 'roi']

            # Prescribed dose and array ,and weight
            pd = self.opt_params.loc[j, 'p_d']
            pd_array = np.full(len(self.ROI_voxel_list[i]), pd)

            weight = self.opt_params.loc[j, 'weight']

            # Calculate dose

            # Time saving for dose calculation:
            # If the dose array is the same as the previous objective/constraint
            # then we dont calculate the dose again
            if j == 0:
                dose, rbe = self.biological_dose(x, i)

            else:
                if i != self.opt_params.loc[j-1, 'roi']:
                    dose, rbe = self.biological_dose(x, i)

            # Mean dose objective
            if self.opt_params.loc[j, 'opt_type'] == 1:

                up = np.square(pd_array - dose) * weight
                down = pd**2
                cost += np.sum(up / down)

            # Max dose constraint
            elif self.opt_params.loc[j, 'opt_type'] == 2:
                dose_np = np.array(dose.toarray()).flatten()

                up = np.square(pd_array - dose_np) * weight
                down = pd**2
                temp_cost = up / down

                temp_cost[dose_np < pd] = 0  # Heaviside function
                cost += np.sum(temp_cost)
            # Max LET constraint
            elif self.opt_params.loc[j, 'opt_type'] == 3:
                # Calculate LETd
                dose_p = x.dot(self.sep_dose_p_list[i])
                up = x.dot(self.sep_let_times_dosep[i])
                dose_p = sparse_to_numpy(dose_p)
                dose_np = sparse_to_numpy(1.1*x.dot(self.sep_dose_list[i]))
                up = sparse_to_numpy(up)
                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)
                let_d[dose_np < float(self.let_co)] = 0

                # Same cost function as for the dose
                up = np.square(pd_array - let_d) * weight
                down = pd**2
                temp_cost = up / down
                temp_cost[let_d < pd] = 0  # Heaviside function
                cost += np.sum(temp_cost)

            # Minimum LET objective
            elif self.opt_params.loc[j, 'opt_type'] == 4:
                # Calculate LETd
                dose_p = x.dot(self.sep_dose_p_list[i])
                up = x.dot(self.sep_let_times_dosep[i])
                dose_p = sparse_to_numpy(dose_p)
                dose_np = sparse_to_numpy(1.1*x.dot(self.sep_dose_list[i]))
                up = sparse_to_numpy(up)
                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)
                let_d[dose_np < float(self.let_co)] = 0

                # Same cost function as for the dose
                up = np.square(pd_array - let_d) * weight
                down = pd**2
                temp_cost = up / down
                temp_cost[let_d > pd] = 0  # Heaviside function
                cost += np.sum(temp_cost)
            # Minimum dose objective
            elif self.opt_params.loc[j, 'opt_type'] == 5:

                dose_np = np.array(dose.toarray()).flatten()

                up = np.square(pd_array - dose_np) * weight
                down = pd**2
                temp_cost = up / down

                temp_cost[dose_np > pd] = 0  # Heaviside function
                cost += np.sum(temp_cost)

            # Mean dose objective
            elif self.opt_params.loc[j, 'opt_type'] == 6:

                mean_dose = dose.mean()
                if mean_dose >= pd:
                    cost += (dose.shape[1]*weight*pow(pd-mean_dose, 2))/pd

        return cost

    def cost_function_der_sep(self, x):
        """The partial derivative of the cost function, where x is the
        intensity"""
        tic = time.time()
        der = np.zeros(x.size)
        x = scipy.sparse.csr_matrix(x)

        for j in range(len(self.opt_params)):

            # i is the number of the ROI
            i = self.opt_params.loc[j, 'roi']

            # Prescribed dose and array ,and weight
            pd = self.opt_params.loc[j, 'p_d']
            pd_array = np.full(len(self.ROI_voxel_list[i]), pd)

            weight = self.opt_params.loc[j, 'weight']

            # Calculate dose

            # Time saving for dose calculation:
            # If the dose array is the same as the previous objective/constraint
            # then we dont calculate the dose again
            if j == 0:
                dose, rbe = self.biological_dose(x, i)

            else:
                if i != self.opt_params.loc[j-1, 'roi']:
                    dose, rbe = self.biological_dose(x, i)

            if self.opt_params.loc[j, 'opt_type'] == 1:

                a = weight*(-2 * (pd_array - dose)) / (pd**2)

                a = np.array(a).flatten()
                a = sp.csr_matrix(a)

                a.multiply(rbe)
                b = self.sep_dose_list[i].T

                der += sparse_to_numpy(a.dot(b))

            elif self.opt_params.loc[j, 'opt_type'] == 2:

                a = weight*(-2 * (pd_array - dose)) / (pd**2)

                a = np.array(a).flatten()

                dose_np = np.array(dose.toarray()).flatten()

                a[dose_np < pd] = 0
                a = sp.csr_matrix(a)
                a.multiply(rbe)

                b = self.sep_dose_list[i].T

                der += sparse_to_numpy(a.dot(b))

            elif self.opt_params.loc[j, 'opt_type'] == 3:

                dose_p = x.dot(self.sep_dose_p_list[i])
                up = x.dot(self.sep_let_times_dosep[i])
                dose_p = sparse_to_numpy(dose_p)

                dose_np = sparse_to_numpy(1.1*x.dot(self.sep_dose_list[i]))

                up = sparse_to_numpy(up)

                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)

                let_d[dose_np < float(self.let_co)] = 0

                a = weight*(-2 * (pd_array - let_d)) / (pd**2)

                a[let_d < pd] = 0

                let_times_dosep_np = self.sep_let_times_dosep[i].toarray()

                first = let_times_dosep_np / dose_p

                second = up/np.square(dose_p)
                second2 = self.sep_dose_p_list[i].toarray()

                down3 = first-(second*second2)

                a = np.array(a).flatten()
                der += np.dot(a, down3.T)

            elif self.opt_params.loc[j, 'opt_type'] == 4:
                ttic = time.time()
                dose_p = x.dot(self.sep_dose_p_list[i])
                up = x.dot(self.sep_let_times_dosep[i])

                dose_np = sparse_to_numpy(1.1*x.dot(self.sep_dose_list[i]))
                let_d = np.divide(up.data, dose_p.data)

                let_d[dose_np < float(self.let_co)] = 0

                a = weight*(-2 * (pd_array - let_d)) / (pd**2)
                a[let_d > pd] = 0

                let_times_dosep_np = self.sep_let_times_dosep[i].toarray()

                first = let_times_dosep_np / dose_p.data
                second = up.data/np.square(dose_p.data)
                second2 = self.sep_dose_p_list[i].toarray()

                down3 = first-(second*second2)

                a = np.array(a).flatten()
                der += np.dot(a, down3.T)

            elif self.opt_params.loc[j, 'opt_type'] == 5:
                a = weight*(-2 * (pd_array - dose)) / (pd**2)
                a = np.array(a).flatten()
                dose_np = np.array(dose.toarray()).flatten()
                a[dose_np > pd] = 0
                a = sp.csr_matrix(a)
                a.multiply(rbe)
                b = self.sep_dose_list[i].T
                der += sparse_to_numpy(a.dot(b))
            elif self.opt_params.loc[j, 'opt_type'] == 6:

                mean_dose = dose.mean()
                if mean_dose >= pd:
                    rbe_list = self.sep_dose_list[i].multiply(rbe)
                    rbe_list = rbe_list.T

                    scalar = dose.shape[1]*weight * \
                        (-2 * (pd - mean_dose)) / (pd**2)
                    number_of_voxels = dose.shape[1]
                    b = np.array((rbe_list.sum(axis=0))).flatten()
                    der += (b/number_of_voxels)*scalar

        print("Time for der cost: {:.5f}".format(time.time()-tic))
        return der

    def cost_function_ro(self, x):
        """Cost function that we try to minimize.
        The input x is the array of current intensities"""
        tic = time.time()
        cost = 0
        x = scipy.sparse.csr_matrix(x)

        # For every objective/constraint
        for j in range(len(self.opt_params)):
            # i is the number of the ROI
            i = self.opt_params.loc[j, 'roi']

            # Prescribed dose and array ,and weight
            pd = self.opt_params.loc[j, 'p_d']
            pd_array = np.full(len(self.ROI_voxel_list[i]), pd)

            weight = self.opt_params.loc[j, 'weight']

            # Calculate dose
            if self.opt_params.loc[j, 'robust'] == 1:
                index = 1
            else:
                index = 0

            if index == 0:
                stoch_weight = self.stochastic_weighting
            else:
                stoch_weight = 1
            # Time saving for dose calculation:
            # If the dose array is the same as the previous objective/constraint
            # then we dont calculate the dose again
            if j == 0:
                dose, rbe = self.biological_dose(x, i, 1)
            else:
                if i != self.opt_params.loc[j-1, 'roi']:
                    dose, rbe = self.biological_dose(x, i, 1)
                stoch_weight = 1

            # Mean dose objective
            if self.opt_params.loc[j, 'opt_type'] == 1:

                up = np.square(pd_array - dose) * weight
                down = pd**2
                cost += np.sum(up / down)*stoch_weight

            # Max dose constraint
            elif self.opt_params.loc[j, 'opt_type'] == 2:
                dose_np = np.array(dose.toarray()).flatten()

                up = np.square(pd_array - dose_np) * weight
                down = pd**2
                temp_cost = up / down

                temp_cost[dose_np < pd] = 0  # Heaviside function
                cost += np.sum(temp_cost)*stoch_weight
            # Max LET constraint
            elif self.opt_params.loc[j, 'opt_type'] == 3:
                # Calculate LETd
                dose_p = x.dot(self.collected_dose_p_list[index][i])
                up = x.dot(self.collected_let_times_dosep_list[index][i])
                dose_p = sparse_to_numpy(dose_p)
                dose_np = sparse_to_numpy(
                    1.1*x.dot(self.collected_dose_list[index][i]))
                up = sparse_to_numpy(up)
                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)
                let_d[dose_np < float(self.let_co)] = 0

                # Same cost function as for the dose
                up = np.square(pd_array - let_d) * weight
                down = pd**2
                temp_cost = up / down
                temp_cost[let_d < pd] = 0  # Heaviside function
                cost += np.sum(temp_cost)*stoch_weight

            # Minimum LET objective
            elif self.opt_params.loc[j, 'opt_type'] == 4:
                # Calculate LETd
                dose_p = x.dot(self.collected_dose_p_list[index][i])
                up = x.dot(self.collected_let_times_dosep_list[index][i])
                dose_p = sparse_to_numpy(dose_p)
                dose_np = sparse_to_numpy(
                    1.1*x.dot(self.collected_dose_list[index][i]))
                up = sparse_to_numpy(up)
                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)
                let_d[dose_np < float(self.let_co)] = 0

                # Same cost function as for the dose
                up = np.square(pd_array - let_d) * weight
                down = pd**2
                temp_cost = up / down
                temp_cost[let_d > pd] = 0  # Heaviside function
                cost += np.sum(temp_cost)*stoch_weight
            # Minimum dose objective
            elif self.opt_params.loc[j, 'opt_type'] == 5:

                dose_np = np.array(dose.toarray()).flatten()

                up = np.square(pd_array - dose_np) * weight
                down = pd**2
                temp_cost = up / down

                temp_cost[dose_np > pd] = 0  # Heaviside function
                cost += np.sum(temp_cost)*stoch_weight
            elif self.opt_params.loc[j, 'opt_type'] == 6:

                mean_dose = dose.mean()
                if mean_dose >= pd:
                    cost += (dose.shape[1]*weight*pow(pd-mean_dose, 2))/pd
        # print ("Time for cost: {:.5f}".format(time.time()-tic))
        return cost

    def cost_function_der_ro(self, x):
        """The partial derivative of the cost function, where x is the
        intensity"""
        tic = time.time()
        der = np.zeros(x.size)
        x = scipy.sparse.csr_matrix(x)

        for j in range(len(self.opt_params)):

            # i is the number of the ROI
            i = self.opt_params.loc[j, 'roi']

            # Prescribed dose and array ,and weight
            pd = self.opt_params.loc[j, 'p_d']
            pd_array = np.full(len(self.ROI_voxel_list[i]), pd)

            weight = self.opt_params.loc[j, 'weight']

            if self.opt_params.loc[j, 'robust'] == 1:
                index = 1
            else:
                index = 0

            if index == 0:
                stoch_weight = self.stochastic_weighting
            else:
                stoch_weight = 1

            # Calculate dose

            # Time saving for dose calculation:
            # If the dose array is the same as the previous objective/constraint
            # then we dont calculate the dose again
            if j == 0:
                dose, rbe = self.biological_dose(x, i, 1)

            else:
                if i != self.opt_params.loc[j-1, 'roi']:
                    dose, rbe = self.biological_dose(x, i, 1)

            if self.opt_params.loc[j, 'opt_type'] == 1:
                # ttic =time.time()
                a = weight*(-2 * (pd_array - dose)) / (pd**2)

                a = np.array(a).flatten()
                a = sp.csr_matrix(a)
                b = rbe*self.sep_dose_list[i].T

                der += sparse_to_numpy(a.dot(b))*stoch_weight

            elif self.opt_params.loc[j, 'opt_type'] == 2:

                a = weight*(-2 * (pd_array - dose)) / (pd**2)

                a = np.array(a).flatten()

                dose_np = np.array(dose.toarray()).flatten()

                a[dose_np < pd] = 0
                a = sp.csr_matrix(a)

                b = rbe*self.sep_dose_list[i].T

                der += sparse_to_numpy(a.dot(b))*stoch_weight

            elif self.opt_params.loc[j, 'opt_type'] == 3:
                dose_p = x.dot(self.collected_dose_p_list[index][i])
                up = x.dot(self.collected_let_times_dosep_list[index][i])
                dose_p = sparse_to_numpy(dose_p)
                dose_np = sparse_to_numpy(dose)
                up = sparse_to_numpy(up)
                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)
                let_d[dose_np < float(self.let_co)] = 0
                a = weight*(-2 * (pd_array - let_d)) / (pd**2)
                a[let_d < pd] = 0
                let_times_dosep_np = self.collected_let_times_dosep_list[index][i].toarray(
                )
                first = let_times_dosep_np / dose_p
                second = up/np.square(dose_p)
                second2 = self.collected_dose_p_list[index][i].toarray()
                down3 = first-(second*second2)
                a = np.array(a).flatten()
                der += np.dot(a, down3.T)*stoch_weight

            elif self.opt_params.loc[j, 'opt_type'] == 4:
                dose_p = x.dot(self.collected_dose_p_list[index][i])
                up = x.dot(self.collected_let_times_dosep_list[index][i])
                dose_p = sparse_to_numpy(dose_p)
                dose_np = sparse_to_numpy(dose)
                up = sparse_to_numpy(up)
                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)
                let_d[dose_np < float(self.let_co)] = 0
                a = weight*(-2 * (pd_array - let_d)) / (pd**2)
                a[let_d > pd] = 0
                let_times_dosep_np = self.collected_let_times_dosep_list[index][i].toarray(
                )
                first = let_times_dosep_np / dose_p
                second = up/np.square(dose_p)
                second2 = self.collected_dose_p_list[index][i].toarray()
                down3 = first-(second*second2)
                a = np.array(a).flatten()
                der += np.dot(a, down3.T)*stoch_weight
            elif self.opt_params.loc[j, 'opt_type'] == 5:
                a = weight*(-2 * (pd_array - dose)) / (pd**2)
                a = np.array(a).flatten()
                dose_np = np.array(dose.toarray()).flatten()
                a[dose_np > pd] = 0
                a = sp.csr_matrix(a)
                b = rbe*self.sep_dose_list[i].T
                der += sparse_to_numpy(a.dot(b))*stoch_weight
            elif self.opt_params.loc[j, 'opt_type'] == 6:
                mean_dose = dose.mean()
                if mean_dose >= pd:
                    rbe_list = self.sep_dose_list[i].multiply(rbe)
                    rbe_list = rbe_list.T
                    scalar = dose.shape[1]*weight * \
                        (-2 * (pd - mean_dose)) / (pd**2)
                    number_of_voxels = dose.shape[1]
                    b = np.array((rbe_list.sum(axis=0))).flatten()
                    der += (b/number_of_voxels)*scalar
        return der

    def normalize(self, x):
        i = int(self.norm[0])
        dose, rbe = self.biological_dose(x, i)
        dose = sparse_to_numpy(dose)
        median = np.median(dose)
        ratio = (float(self.norm[1])/self.fractions)/median
        return x.multiply(ratio)

    def print_metrics(self, x, n, prev_cost, not_robust=True):
        """Function to print out metrics for each iteration of the optimization.
        Takes in the intensity, old cost, and if we are robustly optimizing"""
        x = ensure_csr_matrix(x)
        # Are we plotting LET stuff
        let_stuff = False
        cost = 0
        goal_table = ["Mean dose", "Maximum dose", "Maximum LETd",
                      "Minimum LETd", "Minimum dose", "Max Mean Dose"]

        # Print the different headlines
        roi_list = ['ROI', 'Goal', 'Value', 'Weight',
                    "Robust", 'Mean', 'Median', 'Max', 'Min', 'Cost', 'New weight (a=2)', 'New weight (a=4)'] # - Erlend
        print(f'{roi_list[0]: <15}|{roi_list[1]: <15}|{roi_list[2]: <5s}|{roi_list[3]: <8s}|{roi_list[4]: <8s}|{roi_list[5]: <9s}|{roi_list[6]: <9s}|{roi_list[7]: <9s}|{roi_list[8]: <9s}|{roi_list[9]: <10s}|{roi_list[10]: <16s}|{roi_list[11]: <10s}')
        # Edited the print and roi_list so it also prints new weight based on the cost-weight relationship of CTV - Erlend 

        first_roi_cost = None # To store the cost of the first ROI, for me this will be mean CTV dose cost - Erlend

        for j in range(len(self.opt_params)):
            # i is the number of the ROI
            i = self.opt_params.loc[j, 'roi']

            # Prescribed dose and array ,and weight
            pd = self.opt_params.loc[j, 'p_d']
            pd_array = np.full(len(self.ROI_voxel_list[i]), pd)

            weight = self.opt_params.loc[j, 'weight']
            if self.opt_params.loc[j, 'robust'] == 1:
                robust = "True"
            else:
                robust = "False"
            # Calculate dose

            # Time saving for dose calculation:
            # If the dose array is the same as the previous objective/constraint
            # then we dont calculate the dose again
            if j == 0:
                dose, rbe = self.biological_dose(x, i)
                dose_np = np.array(dose.toarray()).flatten()
            else:
                if i != self.opt_params.loc[j-1, 'roi']:
                    dose, rbe = self.biological_dose(x, i)
                    dose_np = np.array(dose.toarray()).flatten()

            # Calculate cost and adding them to a total_cost list
            if self.opt_params.loc[j, 'opt_type'] == 1:
                
                up = np.square(pd_array - dose) * weight
                down = pd**2
                cost1 = np.sum(up / down)
                cost += cost1

                first_roi_cost = cost1 # - Erlend 

            # Max dose constraint
            elif self.opt_params.loc[j, 'opt_type'] == 2:

                up = np.square(pd_array - dose) * weight
                down = pd**2
                temp_cost = up / down
                temp_cost = np.array(temp_cost).flatten()
                temp_cost[dose_np < pd] = 0  # Heaviside function
                cost1 = np.sum(temp_cost)
                cost += cost1

            # Max LET constraint
            elif self.opt_params.loc[j, 'opt_type'] == 3:
                # Calculate LETd
                dose_p = x.dot(self.sep_dose_p_list[i])
                up = x.dot(self.sep_let_times_dosep[i])
                dose_p = sparse_to_numpy(dose_p)
                dose_np = sparse_to_numpy(dose)
                up = sparse_to_numpy(up)
                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)
                let_d[dose_np < float(self.let_co)] = 0

                # Same cost function as for the dose
                up = np.square(pd_array - let_d) * weight
                down = pd**2
                temp_cost = up / down
                temp_cost[let_d < pd] = 0  # Heaviside function
                cost1 = np.sum(temp_cost)
                cost += cost1

            # Minimum LET objective
            elif self.opt_params.loc[j, 'opt_type'] == 4:
                # Calculate LETd
                dose_p = x.dot(self.sep_dose_p_list[i])
                up = x.dot(self.sep_let_times_dosep[i])
                dose_p = sparse_to_numpy(dose_p)
                # print (dose)
                dose_np = sparse_to_numpy(dose)
                # print(dose_np)
                up = sparse_to_numpy(up)
                # down = np.dot(x, self.sep_dose_p_list[i])
                let_d = np.divide(up, dose_p)
                let_d = np.nan_to_num(let_d)
                let_d[dose_np < float(self.let_co)] = 0

                # Same cost function as for the dose
                up = np.square(pd_array - let_d) * weight
                down = pd**2
                temp_cost = up / down
                # temp_cost = remove_low_values(temp_cost,let_d,pd)
                temp_cost[let_d > pd] = 0  # Heaviside function
                cost1 = np.sum(temp_cost)
                cost += cost1
            # Minimum dose objective
            elif self.opt_params.loc[j, 'opt_type'] == 5:

                up = np.square(pd_array - dose) * weight
                down = pd**2
                temp_cost = up / down
                temp_cost = np.array(temp_cost).flatten()
                temp_cost[dose_np > pd] = 0  # Heaviside function

                # temp_cost = remove_low_values(temp_cost,dose,pd)
                # temp_cost[dose < pd] = 0 #Heaviside function
                cost1 = np.sum(temp_cost)
                cost += cost1

            elif self.opt_params.loc[j, 'opt_type'] == 6:

                mean_dose = dose.mean()
                if mean_dose >= pd:
                    cost1 = (dose.shape[1]*weight*pow(pd-mean_dose, 2))/pd
                    cost += cost1
                else:
                    cost1 = 0
            # Add all costs to a list if we are not robust optimizing
            if not_robust:
                self.cost_list[j].append(cost1)

            # Used for calculating the relative weight for a criteria compared to the cost-weight relationship of CTV.
            # Only works if CTV is the only ROI with opt type = 1. If opt_params contain more than one version of opt type 1 for CTV, it will
            # calculate based on the last one in the opt_params file.
            weight_cost_2 = ( (first_roi_cost * weight) / (cost1 * 2) ) # Where a = 2 
            weight_cost_4 = ( (first_roi_cost * weight) / (cost1 * 4) ) # Where a = 4

            # Multiply the dose with number of fractions for metrics
            d_met = (dose*self.fractions).toarray()
            # Remove 0 values from the metrics list, rmeove??
            d_met = d_met[d_met != 0]
            # try:

            if self.opt_params.loc[j, 'opt_type'] == 3 or self.opt_params.loc[j, 'opt_type'] == 4:
                l_met = let_d
                l_met = l_met[l_met != 0]
                if not let_stuff:
                    # print('{0:8s}   {1:9s}   {2:11s} {3:9s}   {4:9s}	 {5:9s}'.format('ROI', ' Mean LETd', 'Median LETd','Max LETd', 'Min LETd', 'Cost'))
                    # print (" ### LET ###")
                    let_stuff = True

                print('{0:15s}|{1:15s}|{2:<5.2f}|{3:<8.1f}|{4:<8s}|{5:<9.4f}|{6:<9.4f}|{7:<9.4f}|{8:<9.4f}|{9:<10.4f}|{10:<16.4f}|{11:<9.4f}'.format(self.datfile_list[i][:-4],
                                                                                                                             goal_table[self.opt_params.loc[
                                                                                                                                 j, 'opt_type']-1],
                                                                                                                             self.opt_params.loc[
                                                                                                                                 j, 'p_d'],
                                                                                                                             self.opt_params.loc[
                                                                                                                                 j, 'weight'],
                                                                                                                             robust,
                                                                                                                             np.mean(
                                                                                                                                 l_met),
                                                                                                                             np.median(
                                                                                                                                 l_met),
                                                                                                                             np.max(
                                                                                                                                 l_met),
                                                                                                                             np.min(
                                                                                                                                 l_met),
                                                                                                                             cost1, weight_cost_2, weight_cost_4)) # Added weight_cost - Erlend
            else:
                print('{0:15s}|{1:15s}|{2:<5.2f}|{3:<8.1f}|{4:<8s}|{5:<9.4f}|{6:<9.4f}|{7:<9.4f}|{8:<9.4f}|{9:<10.4f}|{10:<16.4f}|{11:<9.4f}'.format(self.datfile_list[i][:-4],
                                                                                                                             goal_table[self.opt_params.loc[
                                                                                                                                 j, 'opt_type']-1],
                                                                                                                             self.opt_params.loc[
                                                                                                                                 j, 'p_d']*self.fractions,
                                                                                                                             self.opt_params.loc[
                                                                                                                                 j, 'weight'],
                                                                                                                             robust,
                                                                                                                             np.mean(
                                                                                                                                 d_met),
                                                                                                                             np.median(
                                                                                                                                 d_met),
                                                                                                                             np.max(
                                                                                                                                 d_met),
                                                                                                                             np.min(
                                                                                                                                 d_met),
                                                                                                                             cost1, weight_cost_2, weight_cost_4)) # Added weight_cost - Erlend
                let_stuff = False
            # except:
                # print('0 values or other error')
                # print np.matmul(np.square((pre_d[i]-dose))/(p_d[i]**2),b[i].T)
        if self.robust_opt:
            print("Total cost after {0:.0f} iterations for main plan: {1:.2f}	{2:.2f}% difference in cost\n\n".format(
                n, cost, 100*(prev_cost-cost)/prev_cost))
        else:
            print("Total cost after {0:.0f} iterations: {1:.2f}	{2:.2f}% difference in cost\n\n".format(
                n, cost, 100*(prev_cost-cost)/prev_cost))
        return cost

    def write_pb_file(self, name, x):
        """Function that writes a file with the current pb-intensities
        Based on an old format to write it. Should be changed"""
        x = ensure_csr_matrix(x)
        if self.norm != "":
            x = ensure_csr_matrix(x)
            x = self.normalize(x)
        f = open("{}/pb_files/{}".format(self.path, name), "w")
        header = "{} {}\n".format(np.sum(x), x.get_shape()[1])
        f.write(header)
        x = sparse_to_numpy(x)
        for i in range(len(x)):
            text = "{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                i+1, "theta", 0, 0, 0, 0, "B_E", 1, 1, 0, 0, x[i], 100, 0)
            f.write(text)
        f.close()
        return

    def prune(self):
        """Prune plans, not yet implemented"""
        f = open(args.pruning, "r")
        lines = f.readlines()
        for i in range(len(lines)):
            for j in range(len(self.dose_pr_primary)):
                self.dose_pr_primary[j][int(lines[i])] = 0
                self.let_pr_primary[j][int(lines[i])] = 0
                self.dose_proton_pr_primary[j][int(lines[i])] = 0
                self.intensities[int(lines[i])] = 0

        f.close()
        return

    def get_ints_from_file(self, filename):
        """Get intensities from a file instead of original weightings.
        Used for continuing an optimization, based on a pb_x.res file
        From chatGPT"""
        data = []  # List to store the extracted 11th items

        with open(filename, 'r') as file:
            lines = file.readlines()

            # Skip the header (assuming it's the first line)
            lines = lines[1:]

            for line in lines:
                # Split the line into sentences based on space
                sentences = line.strip().split()

                if len(sentences) >= 12:
                    # Extract the 11th item (assuming 0-based indexing)
                    item_11 = sentences[11]
                    data.append(item_11)
            # Convert the data list to a NumPy array
            np_array = np.array(data, dtype=float32)
            return np_array

    def biological_dose(self, x, i, robust_index=0):
        """Function that calculates biological dose for ROI #i with intensity x
        Must include different alpha beta values, TODO
        Also, this is a bad way of doing it in terms of division, Needs
        fixing"""

        if robust_index == 0:
            dose = x.dot(self.sep_dose_list[i])
        else:
            dose = x.dot(self.collected_dose_list[robust_index][i])
        # If we use an RBE of 1.1, we dont have to calculate a lot elsewise
        if self.biological_model == "1.1":
            return dose*1.1, 1.1

        # Proton dose

        # let_d[dose_np < float(self.let_co)] = 0
        # Define some arrays
        RBE_max = np.zeros_like(self.sep_dose_list[i])
        RBE_min = np.zeros_like(self.sep_dose_list[i])
        RBE = np.zeros_like(self.sep_dose_list[i])

        # Calculate LETd
        dose_p = x.dot(self.sep_dose_p_list[i])
        up = x.dot(self.sep_let_times_dosep[i])
        dose_p = sparse_to_numpy(dose_p)
        # print (dose)
        dose_np = sparse_to_numpy(dose)
        # print(dose_np)
        up = sparse_to_numpy(up)
        # down = np.dot(x, self.sep_dose_p_list[i])
        let_d = np.divide(up, dose_p)

        # Because of we divide by zero (fix sometime), we need to convert the
        # infinity-values to zero
        let_d = np.nan_to_num(let_d)

        # Make an alpha-beta array
        ab_array = np.zeros_like(let_d)
        ab_array.fill(self.alpha_beta)

        dose_p = dose_p/1.1
        # Calculate RBE max and min for different models
        # Only MCN valid atm
        if self.biological_model == "MCN":
            # Calculate the biological dose for MCN model
            division_term = np.zeros_like(let_d)
            division_term.fill(0.35605/self.alpha_beta)
            RBE_max = 0.99064 + np.multiply(division_term, let_d)
            RBE_min = 1.1012 - 0.0038703 * sqrt(self.alpha_beta) * let_d
            RBE_max = np.nan_to_num(RBE_max)
            RBE_min = np.nan_to_num(RBE_min)
        if self.biological_model == "CAR":
            division_term = np.zeros_like(let_d)
            division_term.fill(0.4136/self.alpha_beta)
            RBE_max = 0.843 + np.multiply(division_term, let_d)
            division_term = np.zeros_like(let_d)
            division_term.fill(0.016116/self.alpha_beta)
            RBE_min = 1.09 - np.multiply(division_term, let_d)
        if self.biological_model == "ROR":

            RBE_max = np.zeros_like(self.let_pr_primary)
            # print np.mean(RBE_max),len(RBE_max)
            new_ab_array = np.zeros_like(self.let_pr_primary)
            new_ab_array.fill(self.alpha_beta)
            # Save time by only calculating this once during the entire optimization(as this does not change)
            RBE_max = np.where(self.let_pr_primary < 37, 1+(0.578*self.let_pr_primary-0.0808*self.let_pr_primary **
                               2+0.00564*self.let_pr_primary**3-0.0000992*self.let_pr_primary**4)/ab_array, 1+(10.5/ab_array))
            RBE_max = np.multiply(RBE_max, self.dose_proton_pr_primary)

            RBE_max = x@RBE_max
            RBE_max = np.divide(RBE_max, dose_p)
            RBE_min = np.ones_like(let_d)

        # General formula for calculating the RBE based on RBE_max and RBE_min
        first_term = np.divide(np.ones_like(dose_p), (2*dose_p))
        second_term = (np.sqrt((np.square(ab_array)) + 4*self.alpha_beta*dose_p *
                       RBE_max + 4*(np.square(dose_p))*(np.square(RBE_min)))) - ab_array
        RBE = np.multiply(first_term, second_term)
        RBE = np.nan_to_num(RBE)
        RBE = sp.csr_matrix(RBE)

        return RBE.multiply(dose), RBE

    def create_dicom(self, intensities, name):
        """Create DICOM based on new intensity, """
        print("Creating DICOM ...")

        intensities = ensure_csr_matrix(intensities)
        file1 = "npz_data/dose_"+self.npz_file
        file2 = "npz_data/dosep_"+self.npz_file
        file3 = "npz_data/let_times_dosep_"+self.npz_file
        tic = time.time()
        dose_pr_primary = load_npz(file1)
        dose_proton_pr_primary = load_npz(file2)
        let_times_dosep = load_npz(file3)

        print(
            "Time for loading large npz-files for DICOM: {:.3f}".format(time.time()-tic))

        def gridmin(dp, iso, pxs): return (dp-iso-pxs/2.0)/10.0
        def gridmax(xmin, db, pxs): return xmin+(db*pxs)/10.0

        # Find RT dose and RT plan file from DICOM-folder

        for filename in os.listdir(self.DICOM_path):
            # print filename
            if ".dcm" in filename.lower():  # Check whether the file is DICOM
                d = dicom.read_file(self.DICOM_path+"/"+filename, force=True)
                if "SOPClassUID" in d:
                    if d.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.8':  # RTPlan
                        pf = dicom.read_file(self.DICOM_path+"/"+filename)

                    elif d.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.2':  # RT Dose
                        ds = dicom.read_file(self.DICOM_path+"/"+filename)
                        # ds_filename = filename

        # These parameters needs to be found automaticly. Consider reading the
        # dicom file

        xbins = int(ds.Columns)
        ybins = int(ds.Rows)
        zbins = int(ds.NumberOfFrames)
        dose_position = np.array(ds.ImagePositionPatient).astype(np.float16)
        pixel_size = np.array(ds.PixelSpacing).astype(np.float16)
        isocenter_list = []
        for j in range(pf.FractionGroupSequence[0].NumberOfBeams):
            ibs = pf.IonBeamSequence[j]
            isocenter_list.append(
                ibs.IonControlPointSequence[0].IsocenterPosition)

        slice_thickness = float(
            ds.GridFrameOffsetVector[1]-ds.GridFrameOffsetVector[0])
        x_min_dose = gridmin(
            dose_position[0], isocenter_list[0][0], pixel_size[0])
        x_max_dose = gridmax(x_min_dose, xbins, pixel_size[0])
        y_min_dose = gridmin(
            dose_position[1], isocenter_list[0][1], pixel_size[1])
        y_max_dose = gridmax(y_min_dose, ybins, pixel_size[1])
        z_min_dose = gridmin(
            dose_position[2], isocenter_list[0][2], slice_thickness)
        z_max_dose = gridmax(z_min_dose, zbins, slice_thickness)
        #
        #
        x_bin_size = (x_max_dose-x_min_dose)/xbins
        y_bin_size = (y_max_dose-y_min_dose)/ybins
        z_bin_size = (z_max_dose-z_min_dose)/zbins
        #
        x_bin_min = int((self.ROI_dimensions[1][0]-x_min_dose)/x_bin_size)
        y_bin_min = int((self.ROI_dimensions[1][1]-y_min_dose)/y_bin_size)
        z_bin_min = int((self.ROI_dimensions[1][2]-z_min_dose)/z_bin_size)
        # print((self.ROI_dimensions[1][0]-x_min_dose)/x_bin_size,int((self.ROI_dimensions[1][0]-x_min_dose)/x_bin_size))

        x_bin_max = x_bin_min + self.ROI_dimensions[0][0]
        y_bin_max = y_bin_min + self.ROI_dimensions[0][1]
        z_bin_max = z_bin_min + self.ROI_dimensions[0][2]
        if self.bio_opt:
            dose_list = ["dose", "LETd", self.biological_model]
        else:
            dose_list = ["dose"]  # ,"LETd"]

        for l in range(len(dose_list)):

            dicom_dose = np.zeros((xbins*ybins*zbins), dtype="float32")
            if l == 0:
                # opt_dose = np.matmul(intensity,self.dose_pr_primary)
                opt_dose = intensities.dot(dose_pr_primary)
                opt_dose = sparse_to_numpy(opt_dose)
            if l == 1:
                up = intensities@let_times_dosep  # Above the fraction
                # print (up)
                down = intensities@dose_proton_pr_primary  # Below the fraction
                # print (down)
                let_d = up/down  # Achieve the LETd
                let_d = scipy.sparse.csr_matrix(let_d)
                # print (let_d)
                # print (let_d)
                # Dose value for cutoff
                dose = intensities@dose_pr_primary
                dose = sparse_to_numpy(dose)
                let_d = sparse_to_numpy(let_d)

                # Because of we divide by zero (fix sometime), we need to convert the
                # infinity-values to zero
                # let_d = np.nan_to_num(let_d)
                # Removes ny let valye from dose region below cutoff
                # print (dose)
                # print (let_d)
                # let_d = remove_low_values(let_d,dose,float(self.let_co))
                # nonzero_mask = np.array(dose[dose.nonzero()] < float(self.let_co))[0]
                # rows = let_d.nonzero()[0][nonzero_mask]
                # cols = let_d.nonzero()[1][nonzero_mask]
                # let_d[rows, cols] = 0
                opt_dose[dose < float(self.let_co)] = 0

            if l == 2:

                opt_dose = intensities.dot(dose_pr_primary)*1.1
                opt_dose = sparse_to_numpy(opt_dose)
            counter = 0

            for i in range(z_bin_min, z_bin_max):
                for j in range(y_bin_min, y_bin_max):
                    for k in range(x_bin_min, x_bin_max):
                        dicom_dose[k + j*xbins + i*ybins *
                                   xbins] = opt_dose[counter]
                        # testfile.write("{} {} {} {}\n".format(opt_dose[counter],i,j,k))
                        counter += 1

            pixel_array = dicom_dose/(float(ds.DoseGridScaling))

            max_flk = float(np.amax(pixel_array))
            # print (ds)
            max_tps = float(np.amax(ds.pixel_array))

            # Need to scale the values of the pixel_array. This because in most cases
            # max value is 65535 due to 16 bit array elements. Therefore all values have
            # to be scaled accordingly so n_max,y_bin_max,z_bin_maxthat no values oveerides the max allowed value.
            # Eclipse Haukeland uses 32 bit
            # print(max_flk,max_tps)
            scale = float(max_flk/max_tps)

            pixel_array = pixel_array/scale
            # Define values to be integers

            pixel_array = np.rint(pixel_array)

            pixel_array = pixel_array.astype(int)
            # print (float(np.amax(ds.pixel_array)))
            # As a result, the DoseGridScaling must also be scaled
            ds.DoseGridScaling = str(
                round((float(ds.DoseGridScaling)*scale), 14))
            #
            pix_arr = np.array(pixel_array, dtype="uint32")
            # print (float(np.amax(ds.pixel_array)))
            ds.PixelData = pix_arr.tobytes()
            # print (float(np.amax(ds.pixel_array)))
            if l == 2:
                if dose_list[l] == "1.1":
                    ds.SeriesDescription = "FLUKA optimized - RBE{} - Plan: {}".format(
                        dose_list[l], self.path)
                    ds.save_as(
                        "{}/DICOMs/{}_FLK_optimized_RBE{}.dcm".format(self.path, name, dose_list[l]))
                else:
                    ds.SeriesDescription = "FLUKA optimized - {} - Plan: {}".format(
                        dose_list[l], self.path)
                    ds.save_as(
                        "{}/DICOMs/{}_FLK_optimized_{}.dcm".format(self.path, name, dose_list[l]))
            elif l == 1:
                ds.SeriesDescription = "FLUKA optimized - LETd - Plan: {}".format(
                    self.path)
                ds.save_as(
                    "{}/DICOMs/{}_FLK_optimized_{}.dcm".format(self.path, name, dose_list[l]))
            elif l == 0:
                ds.SeriesDescription = "FLUKA optimized - Physical dose - Plan: {}".format(
                    self.path)
                ds.save_as(
                    "{}/DICOMs/{}_FLK_optimized_Phys_dose.dcm".format(self.path, name))
        # TODO:Save DICOM in DICOM-folder as well
        print("Finished creating DICOM")


class Optimizer(QtWidgets.QDialog):

    """What to send over: Name of plan, parameter file, dicom location, npzfile, boolean for bio, model for bio,
    aalpha beta for bio, robust opt, robust eval, pruning, number of iterations, update number.
    Send back, name of plot """

    def __init__(self):
        if args.gui:
            super().__init__()
            # super(Dialog, self).__init__()
            self.ui = demo.Ui_Dialog()

            self.ui.setupUi(self)
            self.ui.LoadParameters.clicked.connect(
                self.set_path_and_opt_params)
            self.ui.DicomLoad.clicked.connect(self.set_dicom_location)
            # self.ui.NPZLoad.clicked.connect(self.set_npz_file)
            # self.ui.LOAD.clicked.connect(self.load)
            self.ui.CalculateInit.clicked.connect(self.calculate_init)
            self.ui.Optimize.clicked.connect(self.optimize)

            self.ui.OptParamsTable.setColumnWidth(0, 195)
            self.ui.OptParamsTable.setColumnWidth(1, 160)
            self.ui.OptParamsTable.setColumnWidth(2, 70)
            self.ui.OptParamsTable.setColumnWidth(3, 200)
            self.ui.OptParamsTable.setColumnWidth(4, 200)

            self.button_group = QButtonGroup()
            self.button_group.addButton(self.ui.button_DosePlot, 1)
            self.button_group.addButton(self.ui.button_LetPlot, 2)
            self.button_group.addButton(self.ui.button_CostPlot, 3)
            # Connect the slot to the buttonToggled signal of the button group
            self.button_group.buttonToggled.connect(
                self.on_radio_button_toggled)

            self.path = args.label

            self.scene = QGraphicsScene()

            # self.plot()
            self.settings = QSettings("Optimizer", "MyApp")
            self.ui.OpenLastSaved.clicked.connect(self.update_settings)
            self.ui.label.addItem(self.path)
            # self.update_settings()

        else:
            # TODO: DEFINE a lot of things here
            self.calculate_init()
            self.optimize()

    def on_radio_button_toggled(self, button, checked):
        if checked:
            if button == self.ui.button_DosePlot:
                self.plot(self.dvh_filename)
            elif button == self.ui.button_LetPlot:
                self.plot(self.lvh_filename)
            elif button == self.ui.button_CostPlot:
                self.plot(self.cost_filename)

    def load(self):
        self.path = self.ui.label.text()
        print("loaded")

    def calculate_init(self):
        # self.path = self.ui.label.text()
        self.number_of_iterations = int(self.ui.iterations.text())
        self.iterations_update = int(self.ui.iterations_update.text())
        self.bio_opt = (self.ui.BioOpt.checkState() == Qt.Checked)
        if self.ui.RobustOpt.checkState() == Qt.Checked:
            self.robust_opt = [
                self.ui.robust_model.currentText(), self.ui.RobustWeight.text()]
        else:
            self.robust_opt = ""
        # (self.ui.RobustEval.checkState()== Qt.Checked)
        self.robust_eval = False
        self.alpha_beta = float(self.ui.AlphaBeta.text())
        self.robust_bio = False
        self.let_co = float(self.ui.let_co.text())
        self.bio_model = self.ui.bio_model.currentText()
        print(self.bio_model)
        self.single = True
        self.parallell = int(self.ui.NumberOfCores.text())
        self.robust_iterations = int(self.ui.RobustIterations.text())
        self.continue_file = ""
        self.eq_weight = (self.ui.EqualWeight.checkState() == Qt.Checked)
        if self.ui.NormButton.checkState() == Qt.Checked:
            self.norm = [self.ui.datfiles.currentIndex(), int(
                self.ui.Normalize.text())]
        else:
            self.norm = ""
        self.fractions = int(self.ui.Fractions.text())
        self.plot_physical_dose = False
        # worker_thread = WorkerThread("init",path,param_file,DICOM_path,npz_file,number_of_iterations,iterations_update,robust_opt,robust_eval,alpha_beta,robust_bio,bio_opt,let_co,bio_opt,single,parallell,robust_iterations,continue_file,plot_physical_dose,opt_param_file,eq_weight,norm)
        self.npz_file = "scored_values.npz"
        self.worker_thread = WorkerThread("init", self.path, self.param_file, self.DICOM_path, self.npz_file, self.number_of_iterations,
                                          self.iterations_update, self.robust_opt, self.robust_eval, self.alpha_beta, self.robust_bio,
                                          self.bio_opt, self.let_co, self.bio_model, self.single, self.parallell,
                                          self.robust_iterations, self.continue_file, self.plot_physical_dose, self.param_file, self.eq_weight, self.norm)
        self.worker_thread.update_dvh.connect(self.plot_dvh)
        self.worker_thread.update_dvh_ro.connect(self.plot_dvh_ro)
        self.worker_thread.load()
        self.worker_thread.calculate_init()

    def optimize(self):
        self.number_of_iterations = int(self.ui.iterations.text())
        self.iterations_update = int(self.ui.iterations_update.text())
        self.worker_thread.set_iterations(
            self.number_of_iterations, self.iterations_update)
        self.worker_thread.update_plot.connect(self.plot_return)

        self.worker_thread.start()

    def plot_return(self, files):
        self.dvh_filename = files[0]
        self.lvh_filename = files[1]
        if self.ui.button_DosePlot.isChecked():
            self.plot(self.dvh_filename)
        else:
            self.plot(self.lvh_filename)

    def plot(self, file):
        if os.path.isfile(file):
            self.scene.clear()
            pixmap = QPixmap(file)  # Load your image using QPixmap
            # pixmap.setFiltering(QPixmap.SmoothTransformation)
            # Set the size of the Scene rectangle?
            self.scene.setSceneRect(0, 0,  840, 510)
            # self.scene.addPixmap(pixmap.scaled(400,400)) #Add a normalised pixel map(0-> 1)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            # pixmap_item.scaled(400,400)

            self.scene.addPixmap(pixmap.scaled(
                840, 510, transformMode=Qt.SmoothTransformation))
            self.ui.CTViewer.fitInView(
                self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.ui.CTViewer.setScene(self.scene)
            self.ui.CTViewer.show()
        else:
            print("Plot file does not exist")

        self.ui.CTViewer.show()

    def plot_dvh(self, figurename, path):

        for i in range(3):
            if i == 1:
                # ax.set_xlabel("LET$_d$[keV/$\mu$m]",size = txtsize) # Fjernet '' -Erlend
                name_of_file = "{}/{}".format(path, figurename[i])
                self.lvh_filename = name_of_file
                # ax.set_xlim([0,6])
            elif i == 0:
                # ax.set_xlabel("Dose [Gy(RBE)]",size = txtsize) # Endret til Dose fra RBE-weighted dose - Erlend
                name_of_file = "{}/{}".format(path, figurename[i])
                self.dvh_filename = name_of_file
                # ax.set_xlim([0,3])
            elif i == 2:
                # ax.set_xlabel("Dose [Gy(RBE)]",size = txtsize)  # Endret til Dose fra RBE-weighted dose - Erlend
                name_of_file = "{}/{}".format(path, figurename[i])
                self.cost_filename = name_of_file
        if self.ui.button_DosePlot.isChecked():
            file = self.dvh_filename
        elif self.ui.button_LetPlot.isChecked():
            file = self.lvh_filename
        else:
            file = self.cost_filename
        # print (file)
        if os.path.isfile(file):
            self.scene.clear()
            pixmap = QPixmap(file)  # Load your image using QPixmap
            # pixmap.setFiltering(QPixmap.SmoothTransformation)
            # Set the size of the Scene rectangle?
            self.scene.setSceneRect(0, 0, 840, 510)
            # self.scene.addPixmap(pixmap.scaled(400,400)) #Add a normalised pixel map(0-> 1)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            # pixmap_item.scaled(400,400)

            self.scene.addPixmap(pixmap.scaled(
                840, 510, transformMode=Qt.SmoothTransformation))
            self.ui.CTViewer.fitInView(
                self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.ui.CTViewer.setScene(self.scene)
            self.ui.CTViewer.show()
        else:
            print("Plot file does not exist")

        self.ui.CTViewer.show()

    def plot_dvh_ro(self, plot_dose_list, plot_volume_list, figurename, path, datfile_list):

        print(len(plot_dose_list))
        print(len(plot_volume_list))

        for k in range(2):
            # fig, ax = plt.subplots()

            # print (len(plot_volume_list),len(plot_volume_list[0]),len(plot_volume_list[0][0]),len(plot_volume_list[0][0][0]),len(plot_volume_list[0][0][0][0]))
            # for i in range(len(plot_dose_list)): #Number of robust scenarios
            # print (plot_dose_list[i])
            # 	if plot_dose_list[i] != None:
            # 		dose_list = plot_dose_list[i][k] #Dose and let list for each robust scenario
            # 		volume_list = plot_volume_list[i][k]
            #
            #
            # 		color_list = ["blue","gray","red",
            # 						"darkgreen", "lime","orangered","peru",
            # 						"darkgoldenrod","khaki","darkkhaki","gold",
            # 						"yellowgreen","greenyellow","lawngreen","green",
            # 						"lightblue","deepskyblue","dodgerblue","blue",
            # 						"violet","fuchsia","deeppink","purple",
            # 						"black","gray","silver","gainsboro","green","yellow","c","g","purple","orange","yellow","teal"]
            # 		r_color_list = ["green","yellow","c","g","purple","orange","yellow","teal"]
            # 		model_names_bio = []
            #
            # 		lw = 5
            # 		alf = 0.8
            #
            # 			#PTV
            # 		linestyles =["-","--","-.",":"]
            # 	#print dose_list[0][1]
            # 		#print (len(dose_list),len(dose_list[0]),len(dose_list[0][0]))
            # 		for j in range(len(dose_list[0])):
            # 			if i == 0:
            # 				a = ax.plot(dose_list[-1][j],volume_list[-1][j],color = color_list[j],linestyle = linestyles[0],label =  "{}".format(datfile_list[j][:-4]),linewidth = lw,alpha = alf)
            # 			else:
            # 				a = ax.plot(dose_list[-1][j],volume_list[-1][j],color = color_list[j],linestyle = linestyles[0],label =  None,linewidth = lw-2,alpha = alf-0.4)
            #
            # 		#print (i,k)
            # case = ["a)","b)"]
            # ROI_list = ["let pr beam","let mean","dose-averaged LET per pb per vox", "dose per pb per vox"]
            # x_lim_table = [[0,6],[0,5.5],[0,70],[0,70]]
            # ax.set_ylabel("Volume [%]",size = txtsize)
            # if k == 1:
            # 	ax.set_xlabel("'LET$_d$[keV/$\mu$m]'",size = txtsize)
            # 	name_of_file = "{}/{}.png".format(path,figurename[k])
            # 	self.lvh_filename = name_of_file
            # 	ax.set_xlim([0,6])
            # else:
            # 	ax.set_xlabel("RBE-weighted dose [Gy(RBE)]",size = txtsize)
            # 	name_of_file = "{}/{}.png".format(path,figurename[k])
            # 	self.dvh_filename = name_of_file
            # 	ax.set_xlim([0,3])
            # ax.grid(color='black',alpha = 0.5)
            #
            # ax.tick_params(axis='both', which='major', labelsize=ticksize)
            # ax.legend(loc='upper right',fancybox=True, shadow=True, ncol=1,fontsize = lgndsize)
            #
            # fig = plt.gcf()
            # left = 0.09
            # right = 0.95
            # bottom = 0.18
            # top = 0.93
            # wspace = 0.15
            # hspace = 0.24
            # fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top, wspace = wspace, hspace = hspace)
            # fig.set_size_inches((28, 17))
            #
            # fig.savefig(name_of_file)
            # plt.close('all')

            if k == 1:

                name_of_file = "{}/{}.png".format(path, figurename[k])
                self.lvh_filename = name_of_file
            else:
                name_of_file = "{}/{}.png".format(path, figurename[k])
                self.dvh_filename = name_of_file

        if self.ui.button_DosePlot.isChecked():
            file = self.dvh_filename
        else:
            file = self.lvh_filename
        # print (self.lvh_filename,self.dvh_filename)
        if os.path.isfile(file):
            self.scene.clear()
            pixmap = QPixmap(file)  # Load your image using QPixmap
            # pixmap.setFiltering(QPixmap.SmoothTransformation)
            # Set the size of the Scene rectangle?
            self.scene.setSceneRect(0, 0, 840, 510)
            # self.scene.addPixmap(pixmap.scaled(400,400)) #Add a normalised pixel map(0-> 1)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            # pixmap_item.scaled(400,400)

            self.scene.addPixmap(pixmap.scaled(
                840, 510, transformMode=Qt.SmoothTransformation))
            self.ui.CTViewer.fitInView(
                self.scene.sceneRect(), Qt.KeepAspectRatio)
            self.ui.CTViewer.setScene(self.scene)
            self.ui.CTViewer.show()
        else:
            print("Plot file does not exist")

        self.ui.CTViewer.show()

    def update_settings(self):

        self.param_file = self.settings.value("last_selected_opt_param")
        self.item = QListWidgetItem((str(self.param_file).split("/")[-1]))
        self.ui.ParamCheck.addItem(self.item)

        self.DICOM_path = self.settings.value("last_selected_DICOM_path")
        self.item = QListWidgetItem((str(self.DICOM_path).split("/")[-1]))
        self.ui.DICOMcheck.addItem(self.item)

        self.fill_tables()
        # self.npz_file = self.settings.value("last_selected_npz_file")
        # self.item = QListWidgetItem(str(self.npz_file).split("/")[-1])
        # self.ui.NPZcheck.addItem(self.item)

    def set_dicom_location(self):
        selected_file, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "All Files (*);;Text Files (*.txt)")
        if selected_file:
            self.DICOM_path = QFileInfo(selected_file).path()
            print("Selected file:", selected_file)
            print("Folder path:", self.DICOM_path)
            self.settings.setValue("last_selected_DICOM_path", self.DICOM_path)
            self.item = QListWidgetItem((str(self.DICOM_path).split("/")[-1]))
            self.ui.DICOMcheck.addItem(self.item)
            # self.param_file = "opt_params.csv"
            return

    def fill_tables(self):
        with open(self.param_file) as f:
            header = f.readline()
            self.datfile_list = f.readline().split(",")
        # Remove the \n from the last element
        self.datfile_list[-1] = self.datfile_list[-1][:-1]
        f.close()
        while ("" in self.datfile_list):
            self.datfile_list.remove("")

        for item in self.datfile_list:
            print(item)
            print(type(item))
            self.ui.datfiles.addItem(item)

        goal_table = ["Mean dose", "Maximum dose", "Maximum LETd",
                      "Minimum LETd", "Minimum dose", "Max Mean Dose"]
        yes_no = ["Yes", "No"]

        print(self.ui.datfiles.currentIndex())
        # Read the rest of the table into a pandas dataframe, which can be used
        # freely later. Can also add whatever here
        self.opt_params = pd.read_csv(self.param_file, sep=',', skiprows=2)
        # print (self.opt_params)

        # df = df.replace(',','', regex=True)
        # Print the op	timization criterias
        fractions = int(self.ui.Fractions.text())

        self.ui.OptParamsTable.setRowCount(self.opt_params.shape[0])
        self.ui.OptParamsTable.setColumnCount(self.opt_params.shape[1])
        print(self.opt_params.shape[0])
        print(self.opt_params.shape[1])
        for j in range(len(self.opt_params)):
            if self.opt_params.loc[j, 'opt_type'] != 3 and self.opt_params.loc[j, 'opt_type'] != 4:

                # print("{} of {} to {} with weight {}".format(table[self.opt_params.loc[j, 'opt_type']-1],round(float(self.opt_params.loc[j, 'p_d']),2),self.datfile_list[self.opt_params.loc[j, 'roi']][:-4],self.opt_params.loc[j, 'weight']))
                # else:
                self.opt_params.loc[j, 'p_d'] /= fractions
            for k in range(5):
                # print (self.opt_params.iloc[j, k])
                if k == 0:
                    # print(self.datfile_list[self.opt_params.iloc[j, k]])
                    item = QTableWidgetItem(
                        self.datfile_list[self.opt_params.iloc[j, k]])
                    self.ui.OptParamsTable.setItem(j, k, item)

                elif k == 3:
                    # print(self.datfile_list[self.opt_params.iloc[j, k]])
                    item = QTableWidgetItem(
                        goal_table[self.opt_params.iloc[j, k]-1])
                    self.ui.OptParamsTable.setItem(j, k, item)

                elif k == 4:
                    # print(self.datfile_list[self.opt_params.iloc[j, k]])
                    item = QTableWidgetItem(yes_no[self.opt_params.iloc[j, k]])
                    self.ui.OptParamsTable.setItem(j, k, item)

                else:
                    # print(self.opt_params.iloc[j, k])
                    item = QTableWidgetItem(str(self.opt_params.iloc[j, k]))
                    self.ui.OptParamsTable.setItem(j, k, item)

        # self.ui.OptParamsTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        # self.ui.OptParamsTable.resizeRowsToContents()

    def set_path_and_opt_params(self):
        if args.gui:
            self.param_file, _ = QFileDialog.getOpenFileName(
                self, "Open File", "", "All Files (*);;Text Files (*.txt)")

            self.item = QListWidgetItem((str(self.param_file).split("/")[-1]))
            self.ui.ParamCheck.addItem(self.item)
            self.settings.setValue("last_selected_opt_param", self.param_file)
            self.settings.setValue("last_selected_label", self.path)
        else:
            self.param_file = "opt_params.csv"
            path = self.path
        self.fill_normalize_table()

    def set_npz_file(self):
        self.npz_file, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "All Files (*);;Text Files (*.txt)")
        self.item = QListWidgetItem(str(self.npz_file).split("/")[-1])
        self.ui.NPZcheck.addItem(self.item)
        self.settings.setValue("last_selected_npz_file", self.npz_file)


def biological_dose(x, dose_pr_primary, dose_proton_pr_primary, let_times_dosep, alpha_beta, biological_model):
    dose = x.dot(dose_pr_primary)
    if biological_model == "1.1":
        return dose*1.1, 1.1

    # Calculate LETd
    dose_p = x.dot(dose_proton_pr_primary)
    up = x.dot(let_times_dosep)
    dose_p = sparse_to_numpy(dose_p)
    # print (dose)
    dose_np = sparse_to_numpy(dose)
    # print(dose_np)
    up = sparse_to_numpy(up)
    # down = np.dot(x, sep_dose_p_list[i])
    let_d = np.divide(up, dose_p)

    # Because of we divide by zero (fix sometime), we need to convert the
    # infinity-values to zero
    let_d = np.nan_to_num(let_d)

    # Make an alpha-beta array
    ab_array = np.zeros_like(let_d)
    ab_array.fill(alpha_beta)

    # Calculate RBE max and min for different models
    # Only MCN valid atm
    if biological_model == "MCN":
        # Calculate the biological dose for MCN model
        division_term = np.zeros_like(let_d)
        division_term.fill(0.35605/alpha_beta)
        RBE_max = 0.99064 + np.multiply(division_term, let_d)
        RBE_min = 1.1012 - 0.0038703 * sqrt(alpha_beta) * let_d
        RBE_max = np.nan_to_num(RBE_max)
        RBE_min = np.nan_to_num(RBE_min)

    # General formula for calculating the RBE based on RBE_max and RBE_min
    first_term = np.divide(np.ones_like(dose_p), (2*dose_p))
    second_term = (np.sqrt((np.square(ab_array)) + 4*alpha_beta*dose_p *
                   RBE_max + 4*(np.square(dose_p))*(np.square(RBE_min)))) - ab_array
    RBE = np.multiply(first_term, second_term)
    RBE = np.nan_to_num(RBE)
    RBE = sp.csr_matrix(RBE)

    return RBE.multiply(dose), RBE


def evaluate_robust(args):
    # global collected_dose_list
    tic = time.time()
    # name, x,datfile_list = args
    # voxel_dose = []
    # print ("HERE2 ... {}".format(name))
    # for i in range(len(datfile_list)):
    # 	dose = load_npz("npz_data/{}_dose_{}".format(i,name))
    # 	voxel_dose.append(x.dot(dose))
    #
    # time.sleep(7)
    #
    # print("Time used for reading and evaluating {}: {:5.2f} s".format(name,time.time()-tic))
    name, x, let_co, ROI_voxel_list, opt_params, ROI_cage_size, path, datfile_list, iteration, alpha_beta, biological_model, fractions, robust_let = args
    # print ("HERE ... {}".format(name))

    dvh_array = []
    lvh_array = []

    for i in range(len(ROI_voxel_list)):
        dose_pr_prim = load_npz("npz_data/{}_dose_{}".format(i, name))
        dose_p_pr_prim = load_npz("npz_data/{}_dose_proton_{}".format(i, name))
        let_times_dosep_pr_prim = load_npz(
            "npz_data/{}_let_times_dosep_{}".format(i, name))
        voxel_dose, rbe = biological_dose(
            x, dose_pr_prim, dose_p_pr_prim, let_times_dosep_pr_prim, alpha_beta, biological_model)
        voxel_dose = sparse_to_numpy(voxel_dose)

        dvh_array.append(voxel_dose*fractions)

        up = x@let_times_dosep_pr_prim  # Above the fraction
        # print (up)
        down = x@dose_p_pr_prim  # Below the fraction

        let_d = up/down  # Achieve the LETd
        let_d = scipy.sparse.csr_matrix(let_d)

        dose, rbe = biological_dose(
            x, dose_pr_prim, dose_p_pr_prim, let_times_dosep_pr_prim, alpha_beta, "1.1")
        # print(dose)
        dose = sparse_to_numpy(dose)
        # print(dose)
        let_d = sparse_to_numpy(let_d)

        let_d[dose < float(let_co)] = 0
        # print (let_d)
        # Add the ROI-letd to the list

        # Add the ROI-letd to the list

        lvh_array.append(let_d)

    final_dose_list = []
    final_volume_list = []
    for l in range(2):
        volume_list = []
        dose_list = []
        if l == 1:
            arr = lvh_array
            let_plot = True
            dvh_filename = "{}/plotting_files/{}{}_{}_lvh.txt".format(
                path, iteration, (name.split("/")[-1][:-4]).replace("scored_values", ""), path)
            metrics_filename = "{}/metrics/{}{}_{}_let_metrics.csv".format(
                path, iteration, (name.split("/")[-1][:-4]).replace("scored_values", ""), path)
            plotting_range = 800
        else:
            arr = dvh_array
            let_plot = False
            dvh_filename = "{}/plotting_files/{}{}_{}_dvh.txt".format(
                path, iteration, (name.split("/")[-1][:-4]).replace("scored_values", ""), path)
            metrics_filename = "{}/metrics/{}{}_{}_dose_metrics.csv".format(
                path, iteration, (name.split("/")[-1][:-4]).replace("scored_values", ""), path)
            plotting_range = 400

        number_of_rois = len(arr)
        volume = []
        dose = []
        v_list = []
        d_list = []
        mf = open(metrics_filename, "w")
        # Write the header of the metrics file
        mf.write("Structure,Volume (cc),Mean dose (Gy),Median dose (Gy),Min dose (Gy),Max dose (Gy),D_0.03cc,D_98% (Gy),D_95% (Gy),D_40% (Gy),D_2% (Gy),V_60Gy (cc),V95%,V107%\n")  # - Erlend
        with open(dvh_filename, "w") as f:
            for j in range(number_of_rois):
                f.write("\n")
                f.write("Structure: {}\n".format(datfile_list[j]))
                f.write("Dose [Gy]\t\tVolume [%]\n")
                a = [float(i) for i in arr[j]]
                get_metrics(mf, a, datfile_list[j], 1.8) # Endret fra 2 til 1.8 - Erlend
                volume = []
                dose = []
                for i in range(plotting_range):
                    if not let_plot:
                        # Appends the volume, times the fraction. Stepsize is 0.01 Gy per fraction
                        dose.append(float(i*0.01*fractions))

                        dose_level = i*0.01*fractions
                    else:
                        dose.append(float(i*0.01))

                        dose_level = i*0.01
                    volume.append(
                        100 * (np.sum(np.fromiter((k > dose_level for k in a), dtype=bool)) / float(len(a))))
                    f.write("{}\t\t{}\n".format(float(i * 0.01), volume[-1]))
                    if volume[-1] == 0:
                        break
                v_list.append(volume)
                d_list.append(dose)

        volume_list.append(v_list)
        dose_list.append(d_list)
        final_dose_list.append(dose_list)
        final_volume_list.append(volume_list)
        mf.close()
    print("Time used for reading and evaluating {}: {:5.2f} s".format(
        name.split("/")[-1], time.time()-tic))
    return final_dose_list, final_volume_list


# name, evaluate, intensities, let_co, ROI_voxel_list, dvh_dict, lvh_dict, opt_params, ROI_bool_matrix, ROI_cage_size, path, datfile_list, iteration, alpha_beta, biological_model,results_queue):#args):
def coll_cost_function(args):
    """You should not include LET in the cost function when finding the scenario with the highest cost, as an unlikely scenario might have a very large LET, but the
    the others wont have, and this will affect the optimization and also which scenario gets chosen."""

    name, x, let_co, ROI_voxel_list, opt_params, ROI_cage_size, path, datfile_list, iteration, alpha_beta, biological_model, fractions, robust_let = args

    optimize_all = False

    if name == "scored_values.npz":
        optimize_all = True

    """Cost function that we try to minimize.
	The input x is the array of current intensities"""
    tic = time.time()
    cost = 0
    x = scipy.sparse.csr_matrix(x)
    # For every objective/constraint
    for j in range(len(opt_params)):
        additional_weight = 1
        optimize_objective = False

        # i is the number of the ROI
        i = opt_params.loc[j, 'roi']

        if name == "scored_values.npz":
            optimize_objective = True
            if len(robust_let) > 1:
                if opt_params.loc[j, 'robust'] == 1:
                    additional_weight = float(robust_let[1])
        elif opt_params.loc[j, 'robust'] == 1:
            optimize_objective = True

        # Prescribed dose and array ,and weight
        pd = opt_params.loc[j, 'p_d']
        pd_array = np.full(len(ROI_voxel_list[i]), pd)

        weight = opt_params.loc[j, 'weight']

        # Calculate dose
        # Pseudo code
        if j == 0:
            dose_pr_prim = load_npz("npz_data/{}_dose_{}".format(i, name))
            dose_p_pr_prim = load_npz(
                "npz_data/{}_dose_proton_{}".format(i, name))
            let_times_dosep_pr_prim = load_npz(
                "npz_data/{}_let_times_dosep_{}".format(i, name))
            dose, rbe = biological_dose(
                x, dose_pr_prim, dose_p_pr_prim, let_times_dosep_pr_prim, alpha_beta, biological_model)

        else:
            if i != opt_params.loc[j-1, 'roi']:
                dose_pr_prim = load_npz("npz_data/{}_dose_{}".format(i, name))
                dose_p_pr_prim = load_npz(
                    "npz_data/{}_dose_proton_{}".format(i, name))
                let_times_dosep_pr_prim = load_npz(
                    "npz_data/{}_let_times_dosep_{}".format(i, name))
                dose, rbe = biological_dose(
                    x, dose_pr_prim, dose_p_pr_prim, let_times_dosep_pr_prim, alpha_beta, biological_model)

        # Mean dose objective
        if opt_params.loc[j, 'opt_type'] == 1 and optimize_objective:

            up = np.square(pd_array - dose) * weight
            # up -= pd
            down = pd**2
            # print (type(up),type(down))
            # sys.exit()
            cost += np.sum(up / down)*additional_weight
        # Max dose constraint
        elif opt_params.loc[j, 'opt_type'] == 2 and optimize_objective:
            # up = np.square(pd_array - dose) * weight
            # down = pd**2
            # temp_cost = up / down
            # temp_cost = remove_low_values(temp_cost,dose,pd)
            # #temp_cost[dose < pd] = 0 #Heaviside function
            # cost += np.sum(temp_cost)
            dose_np = np.array(dose.toarray()).flatten()

            up = np.square(pd_array - dose_np) * weight
            down = pd**2
            temp_cost = up / down
            # print (type(temp_cost))
            # print(temp_cost)
            # sys.exit()
            # dose_np = np.array(dose.toarray()).flatten()
            temp_cost[dose_np < pd] = 0  # Heaviside function
            cost += np.sum(temp_cost)*additional_weight
        # Max LET constraint
        elif opt_params.loc[j, 'opt_type'] == 3 and optimize_objective:
            # Calculate LETd
            dose_p = x.dot(dose_p_pr_prim)
            up = x.dot(let_times_dosep_pr_prim)
            dose_p = sparse_to_numpy(dose_p)
            # print (dose)
            dose_np = sparse_to_numpy(1.1*x.dot(dose_pr_prim))
            # print(dose_np)
            up = sparse_to_numpy(up)
            # down = np.dot(x, sep_dose_p_list[i])
            let_d = np.divide(up, dose_p)
            let_d = np.nan_to_num(let_d)
            let_d[dose_np < float(let_co)] = 0

            # Same cost function as for the dose
            up = np.square(pd_array - let_d) * weight
            down = pd**2
            temp_cost = up / down
            # temp_cost = remove_low_values(temp_cost,let_d,pd)
            temp_cost[let_d < pd] = 0  # Heaviside function
            cost += np.sum(temp_cost)*additional_weight

        # Minimum LET objective
        elif opt_params.loc[j, 'opt_type'] == 4 and optimize_objective:
            # Calculate LETd
            dose_p = x.dot(dose_p_pr_prim)
            up = x.dot(let_times_dosep_pr_prim)
            dose_p = sparse_to_numpy(dose_p)
            # print (dose)
            dose_np = sparse_to_numpy(1.1*x.dot(dose_pr_prim))
            # print(dose_np)
            up = sparse_to_numpy(up)
            # down = np.dot(x, sep_dose_p_list[i])
            let_d = np.divide(up, dose_p)
            let_d = np.nan_to_num(let_d)
            let_d[dose_np < float(let_co)] = 0

            # Same cost function as for the dose
            up = np.square(pd_array - let_d) * weight
            down = pd**2
            temp_cost = up / down
            # temp_cost = remove_low_values(temp_cost,let_d,pd)
            temp_cost[let_d > pd] = 0  # Heaviside function
            cost += np.sum(temp_cost)*additional_weight
        # Minimum dose objective
        elif opt_params.loc[j, 'opt_type'] == 5 and optimize_objective:

            dose_np = np.array(dose.toarray()).flatten()

            up = np.square(pd_array - dose_np) * weight
            down = pd**2
            temp_cost = up / down
            # print (type(temp_cost))
            # print(temp_cost)
            # sys.exit()
            # dose_np = np.array(dose.toarray()).flatten()
            temp_cost[dose_np > pd] = 0  # Heaviside function
            cost += np.sum(temp_cost)*additional_weight

        elif opt_params.loc[j, 'opt_type'] == 6 and optimize_objective:

            mean_dose = dose.mean()
            if mean_dose >= pd:
                cost += (dose.shape[1]*weight*pow(pd-mean_dose, 2))/pd

    # print ("Time for cost: {:.5f}".format(time.time()-tic))
    return cost


def coll_cost_function_der(args):
    """The partial derivative of the cost function, where x is the
    intensity"""
    name, x, let_co, ROI_voxel_list, opt_params, ROI_cage_size, path, datfile_list, iteration, alpha_beta, biological_model, fractions, robust_let = args
    tic = time.time()
    der = np.zeros(x.size)
    x = scipy.sparse.csr_matrix(x)

    optimize_all = True

    for j in range(len(opt_params)):
        optimize_objective = False
        # i is the number of the ROI
        additional_weight = 1

        i = opt_params.loc[j, 'roi']

        # Prescribed dose and array ,and weight
        pd = opt_params.loc[j, 'p_d']
        pd_array = np.full(len(ROI_voxel_list[i]), pd)

        weight = opt_params.loc[j, 'weight']

        # Calculate dose

        if name == "scored_values.npz":
            optimize_objective = True
            if len(robust_let) > 1:
                if opt_params.loc[j, 'robust'] == 1:
                    additional_weight = float(robust_let[1])
        elif opt_params.loc[j, 'robust'] == 1:
            optimize_objective = True

        # Time saving for dose calculation:
        # If the dose array is the same as the previous objective/constraint
        # then we dont calculate the dose again

        # Pseudo code

        if j == 0:
            dose_pr_prim = load_npz("npz_data/{}_dose_{}".format(i, name))
            dose_p_pr_prim = load_npz(
                "npz_data/{}_dose_proton_{}".format(i, name))
            let_times_dosep_pr_prim = load_npz(
                "npz_data/{}_let_times_dosep_{}".format(i, name))
            dose, rbe = biological_dose(
                x, dose_pr_prim, dose_p_pr_prim, let_times_dosep_pr_prim, alpha_beta, biological_model)

        else:
            if i != opt_params.loc[j-1, 'roi']:
                dose_pr_prim = load_npz("npz_data/{}_dose_{}".format(i, name))
                dose_p_pr_prim = load_npz(
                    "npz_data/{}_dose_proton_{}".format(i, name))
                let_times_dosep_pr_prim = load_npz(
                    "npz_data/{}_let_times_dosep_{}".format(i, name))
                dose, rbe = biological_dose(
                    x, dose_pr_prim, dose_p_pr_prim, let_times_dosep_pr_prim, alpha_beta, biological_model)

        if opt_params.loc[j, 'opt_type'] == 1 and optimize_objective:
            a = weight*(-2 * (pd_array - dose)) / (pd**2)

            a = np.array(a).flatten()
            a = sp.csr_matrix(a)
            a = a.multiply(rbe)
            b = dose_pr_prim.T

            der += sparse_to_numpy(a.dot(b))*additional_weight

        elif opt_params.loc[j, 'opt_type'] == 2 and optimize_objective:

            a = weight*(-2 * (pd_array - dose)) / (pd**2)

            a = np.array(a).flatten()

            dose_np = np.array(dose.toarray()).flatten()

            a[dose_np < pd] = 0
            a = sp.csr_matrix(a)
            a = a.multiply(rbe)
            b = dose_pr_prim.T

            der += sparse_to_numpy(a.dot(b))*additional_weight

        elif opt_params.loc[j, 'opt_type'] == 3 and optimize_objective:
            dose_p = x.dot(dose_p_pr_prim)
            up = x.dot(let_times_dosep_pr_prim)
            dose_p = sparse_to_numpy(dose_p)
            dose_np = sparse_to_numpy(1.1*x.dot(dose_pr_prim))
            up = sparse_to_numpy(up)
            let_d = np.divide(up, dose_p)
            let_d = np.nan_to_num(let_d)

            let_d[dose_np < float(let_co)] = 0
            a = weight*(-2 * (pd_array - let_d)) / (pd**2)

            a[let_d < pd] = 0

            let_times_dosep_np = let_times_dosep_pr_prim.toarray()
            first = let_times_dosep_np / dose_p

            second = up/np.square(dose_p)
            second2 = dose_p_pr_prim.toarray()

            down3 = first-(second*second2)

            a = np.array(a).flatten()
            der += np.dot(a, down3.T)*additional_weight

        elif opt_params.loc[j, 'opt_type'] == 4 and optimize_objective:

            dose_p = x.dot(dose_p_pr_prim)
            up = x.dot(let_times_dosep_pr_prim)
            dose_p = sparse_to_numpy(dose_p)

            dose_np = sparse_to_numpy(1.1*x.dot(dose_pr_prim))

            up = sparse_to_numpy(up)

            let_d = np.divide(up, dose_p)
            let_d = np.nan_to_num(let_d)

            let_d[dose_np < float(let_co)] = 0

            a = weight*(-2 * (pd_array - let_d)) / (pd**2)

            a[let_d > pd] = 0

            let_times_dosep_np = let_times_dosep_pr_prim.toarray()
            first = let_times_dosep_np / dose_p

            second = up/np.square(dose_p)
            second2 = dose_p_pr_prim.toarray()
            down3 = first-(second*second2)
            a = np.array(a).flatten()

            der += np.dot(a, down3.T)*additional_weight

        elif opt_params.loc[j, 'opt_type'] == 5 and optimize_objective:

            a = weight*(-2 * (pd_array - dose)) / (pd**2)

            a = np.array(a).flatten()

            dose_np = np.array(dose.toarray()).flatten()

            a[dose_np > pd] = 0

            a = sp.csr_matrix(a)
            a = a.multiply(rbe)
            b = dose_pr_prim.T

            der += sparse_to_numpy(a.dot(b))*additional_weight
        elif opt_params.loc[j, 'opt_type'] == 6 and optimize_objective:

            mean_dose = dose.mean()
            if mean_dose >= pd:
                rbe_list = dose_p_pr_prim.multiply(rbe)
                rbe_list = rbe_list.T

                scalar = dose.shape[1]*weight*(-2 * (pd - mean_dose)) / (pd**2)
                number_of_voxels = dose.shape[1]
                b = np.array((rbe_list.sum(axis=0))).flatten()
                der += (b/number_of_voxels)*scalar
    return der


# name, evaluate, intensities, let_co, ROI_voxel_list, dvh_dict, lvh_dict, opt_params, ROI_bool_matrix, ROI_cage_size, path, datfile_list, iteration, alpha_beta, biological_model,results_queue):#args):
def read_npz_files(args):
    """You should not include LET in the cost function when finding the scenario with the highest cost, as an unlikely scenario might have a very large LET, but the
    the others wont have, and this will affect the optimization and also which scenario gets chosen."""
    name, datfile_list = args

    sep_dose_list = np.empty(len(datfile_list), dtype=object)
    sep_dose_p_list = np.empty(len(datfile_list), dtype=object)
    sep_let_times_dosep = np.empty(len(datfile_list), dtype=object)
    # First check if the original .npz file exists.

    tic = time.time()
    # print ("Reading numpy file ... {}".format(name))
    # print (len(datfile_list))
    # Read the data from the files
    for i in range(len(datfile_list)):
        if os.path.isfile(("npz_data/{}_dose_{}".format(i, name))):
            sep_dose_list[i] = load_npz("npz_data/{}_dose_{}".format(i, name))
            # print ("Finished dose for {}".format(name))
        else:
            print("FILE DOES NOT EXIST")
            return None, None, None

        if os.path.isfile(("npz_data/{}_dose_proton_{}".format(i, name))):
            sep_dose_p_list[i] = load_npz(
                "npz_data/{}_dose_proton_{}".format(i, name))
            # print ("Finished dose_p for {}".format(name))
        else:
            print("FILE DOES NOT EXIST")
            return None, None, None

    for i in range(len(datfile_list)):

        if os.path.isfile(("npz_data/{}_let_times_dosep_{}".format(i, name))):
            sep_let_times_dosep[i] = load_npz(
                "npz_data/{}_let_times_dosep_{}".format(i, name))
            # print ("Finished let for {}".format(name))
        else:
            print("FILE DOES NOT EXIST")
            return None, None, None

    return sep_dose_list, sep_dose_p_list, sep_let_times_dosep


def find_voxel_volume(structure_dat):
    """ Reads the voxel sizes that the ROI_datfiles contains for a given structure file.
    Note 'structure_file' should be the structrue file located in ROI_datfiles.
    For example 'CTV.dat'. - Erlend
    """
    with open(structure_dat, "r") as file:
        # Skip the first line
        next(file)

        # Read the bin sizes
        bin_size = next(file).strip()
        bin_x, bin_y, bin_z = map(float, bin_size.split())

        # Read the third line
        line_min = next(file).strip()
        x_m, y_m, z_m = map(float, line_min.split())

        # Read the fourth line
        line_max = next(file).strip()
        x_p, y_p, z_p = map(float, line_max.split())

        voxel_width = (x_p - x_m) / bin_x
        voxel_height = (y_p - y_m) / bin_y
        voxel_depth = (z_p - z_m) / bin_z

        voxel_volume = voxel_width * voxel_height * voxel_depth

    return voxel_volume

def check_voxel(sorted_np_array, voxel_dose, voxel_size):
    """ Suggestion for further work on D_0.03cc metric
    This function will check if the voxel with voxel dose, voxel_dose,
    is inside or partly inside the structure. If fully inside, it will return inside_fraction = 1.
    If partly inside it will return inside_fraction = the fraction of the voxel which is partly inside.

    I think sorted_np_array and voxel_dose will be needed for keeping track of which voxel is being investegated.
    However, it might be neccesary for a voxel_position and a structure_surface function aswell to keep track
    of the position of the surface boundary and if the voxel position lies on this boundary.

    voxel_size will be needed to figure out the fraction of the voxel which lies on the surface boundary.
    - Erlend
    """
    inside_fraction = 1
    return inside_fraction

def get_metrics(mf, arr, struct, prescribed_dose):
    """ Function to get metrics for dose and LET. Takes in the metrics file (mf),
    array of dose or let for each voxel (arr), structure name (struct) and the 
    prescribed_dose.
    Changed most of this function. - Erlend
    """
    # Changed to decending order (e.g 2, 1.9, 1.8)
    sorted_array = sorted(arr, reverse=True)
    
    # Get the total number of voxels for the given structure (struct)
    total_vox = len(sorted_array)
    
    # Get the voxel size (volume in cc)
    voxel_size = find_voxel_volume("ROI_datfiles/"+struct)
    
    # Find the total volume
    total_volume = total_vox * voxel_size

    # Make it into a numpy array
    sorted_array = np.array(sorted_array)

    # Find the number of voxels receiving 60 Gy or more after 30 fractions
    voxel_count = np.sum(sorted_array >= 60)

    # Multiply the number of voxels with the voxel size in cc to get the V_60
    v_60 = voxel_count * voxel_size

    # Define standard metrics
    mean = np.mean(sorted_array)
    median = np.median(sorted_array)
    maximum = np.max(sorted_array)
    minimum = np.min(sorted_array)

    # Calculate D_0.03cc
    d_003cc = 0

    # First find the number of voxels that makes 0.03cm³
    amount_of_voxels = (0.03 / voxel_size) 
    # amount_of_voxels is most likely a float, thus its a whole number of voxels, pluss a fraction of one voxel
    whole_voxel = int(amount_of_voxels)
    frac_voxel = amount_of_voxels - whole_voxel

    # Loop through all voxels that sums up to be 0.03cm³
    for i in range(whole_voxel + 1):
        # If it's the fraction voxel the dose in that voxel will be multipied 
        # by the % of the volume that the fraction constitutes.
        if i == whole_voxel:
            weighting_factor = (frac_voxel / amount_of_voxels)
            dose_contribution = (sorted_array[i] * weighting_factor)
            d_003cc += dose_contribution
        else:
        # Here we use (1 / amount_of_voxels) since its one full voxel dose to account for. 
            weighting_factor = (1 / amount_of_voxels)
            dose_contribution = (sorted_array[i] * weighting_factor)
            d_003cc += dose_contribution
    # For further work: The voxels that recivies the highest dose might be in the boundary of the structure. Thus, the 'amount_of_voxels' is wrong in that scenario.
    # To fix this I propose a check function that checks how much of each voxel is inside the structure (See def check_voxel).
    # Then add that voxel and its 'inside-fraction' to a list. Then keep adding voxels and inside-fractions until
    # (sum of inside-fractions) * voxel_size = 0.03cc. Then to find the D_0.03cc metric it should be a for-loop
    # that loops through the amount of voxels that sums up to be 0.03cc and multiply that by their volume contribution and dose.
    # For example if 2/3 of a voxel is inside the structure and that voxel recieves 54 Gy,
    # the contribution from this voxel would be ( 0.03 / (2/3 * voxel_size) ) * 54 Gy.

    # Additional percentile metrics. Creates a list of the different percentile metrics.
    percent_list = np.percentile(sorted_array, [100-98, 100-95, 100-40, 100-2]) # Add more if needed. NB: if you want D40% you have to do 100-40.
    # Note: Remember to change the mf.write if more percentiles are added. Search for '# Write the header of the metrics file'.


    # Changed the way to calculate the V95% and V107%
    frac = 30 # Change this if another amount of fractions is used
    v_95_percent = ( np.sum(sorted_array >= (prescribed_dose * frac * 0.95)) / total_vox ) * 100
    v_107_percent = ( np.sum(sorted_array >= (prescribed_dose * frac * 1.07)) / total_vox ) * 100

    # Write the standard metrics. Changed it to contain all metrics in one write for better overview
    # The metrics is as follows: Structure name, total volume (cc), mean dose, median dose,
    # minium dose, maxium dose, D_0.03cc, D98%, D95%, D40%, D2%, V_60Gy, V95%, V107%
    metrics = [struct[:-4], total_volume, mean, median, minimum, maximum, d_003cc, percent_list[0], percent_list[1], percent_list[2], percent_list[3], v_60, v_95_percent, v_107_percent]

    mf.write("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(*metrics))
    
    mf.write("\n")


def ensure_csr_matrix(matrix):
    if sp.isspmatrix_csr(matrix):
        # If the matrix is already a CSR matrix, return it as is
        return matrix
    else:
        # Convert the matrix to a CSR matrix
        return sp.csr_matrix(matrix)


def cost_function(args):
    """Cost function that we try to minimize.
    The input x is the array of current intensities"""
    j, opt_params, x, sep_dose_list, sep_dose_p_list, sep_let_times_dosep, let_co, alpha_beta, biological_model, ROI_voxel_list = args
    j = int(j)
    tic = time.time()

    # i is the number of the ROI
    i = opt_params.loc[j, 'roi']

    # Prescribed dose and array ,and weight
    pd = opt_params.loc[j, 'p_d']
    pd_array = np.full(len(ROI_voxel_list[i]), pd)

    weight = opt_params.loc[j, 'weight']

    # Calculate dose

    # Time saving for dose calculation:
    # If the dose array is the same as the previous objective/constraint
    # then we dont calculate the dose again
    dose = biological_dose(x, sep_dose_list, sep_dose_p_list,
                           sep_let_times_dosep, alpha_beta, biological_model, i)

    # Mean dose objective
    if opt_params.loc[j, 'opt_type'] == 1:

        up = np.square(pd_array - dose) * weight
        down = pd**2
        return np.sum(up / down)

    # Max dose constraint
    elif opt_params.loc[j, 'opt_type'] == 2:
        up = np.square(pd_array - dose) * weight
        down = pd**2
        temp_cost = up / down
        temp_cost[dose < pd] = 0  # Heaviside function
        return np.sum(temp_cost)

    # Max LET constraint
    elif opt_params.loc[j, 'opt_type'] == 3:
        # Calculate LETd
        up = np.dot(x, sep_let_times_dosep[i])
        down = np.dot(x, sep_dose_p_list[i])
        let_d = np.divide(up, down)
        let_d = np.nan_to_num(let_d)

        # LET cutoff
        let_d[dose < float(let_co)] = 0

        # Same cost function as for the dose
        up = np.square(pd_array - let_d) * weight
        down = pd**2
        temp_cost = up / down
        temp_cost[let_d < pd] = 0  # Heaviside function
        return np.sum(temp_cost)

    # Minimum LET objective
    elif opt_params.loc[j, 'opt_type'] == 4:
        # Calculate LETd
        up = np.dot(x, sep_let_times_dosep[i])
        down = np.dot(x, sep_dose_p_list[i])
        let_d = np.divide(up, down)
        let_d = np.nan_to_num(let_d)

        # LET cutoff
        let_d[dose < float(let_co)] = 0

        # Same cost function as for the dose
        up = np.square(pd_array - let_d) * weight
        down = pd**2
        temp_cost = up / down
        temp_cost[let_d > pd] = 0  # Heaviside function
        return np.sum(temp_cost)
    # Minimum dose objective
    elif opt_params.loc[j, 'opt_type'] == 5:

        up = np.square(pd_array - dose) * weight
        down = pd**2
        temp_cost = up / down
        temp_cost[dose > pd] = 0  # Heaviside function
        return np.sum(temp_cost)


def sparse_to_numpy(a):
    """Converts a 1d sparse matrix to a 1d numpy ndarray"""
    return np.array(a.toarray()).flatten()


class LogPrintsToFile:
    def __init__(self, log_file_name):
        self.log_file = open(log_file_name, "a")
        self.stdout_original = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.log_file.write(text)
        self.stdout_original.write(text)

    def __del__(self):
        sys.stdout = self.stdout_original
        self.log_file.close()

    def flush(self):
        pass  # Override the flush method to prevent AttributeError


def handle_interrupt(signal, frame):
    # Clean up when Ctrl+C is pressed
    global log_printer
    del log_printer
    sys.exit(1)


Nfeval = 1
tic = time.time()
if __name__ == "__main__":
    if args.label == "":
        print("No name label chosen, terminating")
        sys.exit()
    isExist = os.path.exists(args.label)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(args.label)
    log_file_name = "{}/Opt_log.log".format(args.label)
    if os.path.exists(log_file_name):
        os.remove(log_file_name)

    # log_file_name = f"print_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_printer = LogPrintsToFile(log_file_name)
    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        if args.gui:
            app = QtWidgets.QApplication(sys.argv)
            window = Optimizer()
            window.show()
            sys.exit(app.exec_())
        else:

            path = args.label
            param_file = "opt_params.csv"
            DICOM_path = args.dicom
            npz_file = args.npzfile
            pruning = False
            number_of_iterations = int(args.iterations)
            iterations_update = int(args.iterations_update)
            bio_opt = args.bio_opt
            robust_opt = args.robust_opt
            robust_eval = args.robust_eval
            alpha_beta = args.alpha_beta_ratio
            robust_bio = args.robust_bio
            let_co = args.let_cutoff
            plot_physical_dose = args.plot_physical_dose
            opt_param_file = args.opt_param_file

            single = args.plot_single
            parallell = args.parallell
            robust_iterations = args.robust_iterations
            continue_file = args.continue_file
            eq_weight = args.equal_weighting
            norm = args.normalize

            worker_thread = WorkerThread("init", path, param_file, DICOM_path, npz_file, number_of_iterations, iterations_update, robust_opt, robust_eval, alpha_beta,
                                         robust_bio, bio_opt, let_co, bio_opt, single, parallell, robust_iterations, continue_file, plot_physical_dose, opt_param_file, eq_weight, norm)
            tic = time.time()
            worker_thread.load()
            worker_thread.calculate_init()
            print("Total loading and initialization time: {:4.2f} s".format(
                time.time()-tic))
            worker_thread.optimize()
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully

    # Ensure the log file is closed when the program exits
    del log_printer
    toc = time.time()
    print("Total time: {:4.2f} s".format(toc-tic))


# plot_dvh(name_let,"test_overoptimization.png")
# calculate_init("init.txt",["zPTV_cyl.dat","BrainStem.dat"],"plot_dose.png","zPTV.npz")
# plot_dvh("data2.txt","letd_vs_pbnumber2.png")
# plot_dvh(DDO_YH_let,"DDO_YH.png")
# plot_dvh(PG_HH_let,"pg_HH.png")
# plot_dvh(PG_YH_let,"pg_YH.png")
# plot_scatter()
# plot_dvh_gui()
