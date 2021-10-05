import argparse
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ..data_process import automate_scaling
from ..data_process import corr_a_method


def auto_scale():
    description = '''
    auto_scale: Automatic scaling escan and tscan data using
    ``A-method''
    '''
    epilog = '''
    *Note:
    auto_scale assume each escan have same energy range and
    each tscan have same time range.
    Also it assumes energy unit of
    escan data is KeV but assumes energy unit of tscan_energy_file
    is eV. Moreover energy unit of scaled_escan is eV.
    However time unit for tscan data and escan time must be same.
    if you want to know of further information about stage:
    type auto_scale -1 or type ``TRXASprefitpack_info auto_scale
    '''

    stage_m1 = '''
    stage -1: description
    It prints the description about each stages and aborts.
    For every stage except -1, requires
    prefix, num_of_escan, escan_time, num_of_tscan,
    tscan_energy, time_zeros
    '''
    stage0 = '''
    stage 0: init scaling

    Additionally requires
    ref_escan_index, tscan_index_to_scale, parm_A

    the program read escan_data from
    prefix_escan_1.txt,...,prefix_num_escan.txt
    and tscan_data from
    prefix_tscan_1.txt,...,prefix_tscan_num_tscan.txt
    Also, it read file for parameter A generated by
    fit static.
    Then it fits scaling of escan_data and
    tscan_i_1,...,tscan_i_N to escan_e_ref.
    Now it generates
    prefix_escan_scaled.txt (energy unit: eV)
    prefix_escan_eps_scaled.txt
    prefix_tscan_scaled.txt
    prefix_tscan_eps_scaled.txt
    prefix_A_ref.txt
    If you do not include tscan j for scaling.
    You can see (j+1) th column of prefix_tscan_scaled.txt and
    j th column of tscan_scaled_eps.txt are filled with zeros.
    '''

    stage1 = '''
    stage 1: Correction

    Additionally requires
    ref_escan_index, tscan_index_to_scale

    Note.
    ref_escan_index and tscan_index_to_scale must be set to
    same as stage 0


    See ``corr_a_method`` to see why correction step is needed
    the program read scaled escan data and tscan data from
    prefix_escan_scaled.txt, prefix_escan_eps_scaled.txt,
    prefix_tscan_scaled.txt, prefix_tscan_eps_scaled.txt,
    and then it corrects scaling of escan using tscan_i_1
    It regenerates prefix_escan_scaled.txt and
    prefix_escan_eps_scaled.txt
    '''

    stage2 = '''
    stage 2: further scaling

    Additionally requires
    tscan_index_to_scale

    the program read scaled escan data and tscan data from
    prefix_escan_scaled.txt, prefix_escan_eps_scaled.txt,
    prefix_tscan_scaled.txt, prefix_tscan_eps_scaled.txt,
    and then it fits scaling of tscan_i'_1,...,tscan_i'_N'
    to escan_e_ref'.
    (Prime means i_1,...,i_N and e_ref values are different
     from stage 0)
    Then it regenerates all prefix_*.txt except prefix_A_ref.txt

    Note in this stage do not need to give the file name for
    parameter A.
    '''

    stage3 = '''
    stage 3: sanity check

    In this stage, the program assume, every tscan data are
    scaled to escan data. For sanity check, it gives
    a graph for you.
    '''

    stage4 = '''
    stage 4: Scaling with another tscan data set
    Scale Another set of tscan data using already scaled
    escan data

    Additionally requires
    tscan_index_to_scale

    Before proceed stage 4, you should move
    prefix_tscan_scaled.txt, prefix_tscan_eps_scaled.txt
    and prefix_tscan_xxx.txt to some backup folder and
    rename your another tscan data set to prefix_tscan_xxx.txt
    Also you should give time_zero and energy for such tscan.

    In this stage, it reads scaled escan data and eps from
    prefix_escan_scaled.txt and prefix_escan_eps_scaled.txt
    Then it reads unscaled tscan data and eps from
    prefix_tscan_*.txt
    Next, it scales tscan datas just like stage 1.
    After stage4 finished you should go to stage 2 and stage 3.
    '''

    stage_description = dict()
    stage_description['-1'] = stage_m1
    stage_description['0'] = stage0
    stage_description['1'] = stage1
    stage_description['2'] = stage2
    stage_description['3'] = stage3
    stage_description['4'] = stage4

    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog)
    parser.add_argument("stage", choices=['-1', '0', '1', '2', '3', '4'],
                        help="current stage, " +
                        "set stage to -1 get detailed description")
    parser.add_argument("-p", "--prefix",
                        help="prefix for both escan and tscan file, " +
                        "it will read prefix_escan_i.txt " +
                        "and prefix_tscan_j.txt")
    parser.add_argument("-ne", "--num_of_escan",
                        help="the number of escan files",
                        type=int)
    parser.add_argument("-et", "--escan_time",
                        help="filename for escan delay times (unit: ps)")
    parser.add_argument("-re", "--ref_escan_index",
                        help="index of escan used to " +
                        "the reference for scaling",
                        type=int)
    parser.add_argument("-nt", "--num_of_tscan",
                        help="the number of tscan files",
                        type=int)
    parser.add_argument("-te", "--tscan_energy",
                        help="filename for tscan energy (unit: eV)")
    parser.add_argument("-t0", "--time_zeros",
                        help="filename for time zero of each tscan (unit: ps)")
    parser.add_argument("-ti", "--tscan_index_to_scale", nargs='+',
                        type=int,
                        help="tscan index to scale, " +
                        "use blank separation for multiple arguments")
    parser.add_argument("-a", "--parm_A",
                        help="filename for the parameter A obtained from " +
                        "fit_static")

    args = parser.parse_args()
    args_dict = vars(args)
    stage = args.stage
    if stage == '-1':
        for key in stage_description.keys():
            print(stage_description[key])
        return
    else:

        tmp = ['prefix', 'num_of_escan', 'escan_time',
               'num_of_tscan', 'tscan_energy', 'time_zeros']

        for i in tmp:
            if args_dict[i] is None:
                print(f"{i} is required for stage {stage}!\n")
                return

        print(stage_description[stage])

        prefix = args.prefix
        num_of_escan = args.num_of_escan
        escan_time = np.genfromtxt(args.escan_time)
        num_of_tscan = args.num_of_tscan
        tscan_energy = np.genfromtxt(args.tscan_energy)
        time_zeros = np.genfromtxt(args.time_zeros)

    if stage not in ['3', '4']:
        e = np.genfromtxt(f'{prefix}_escan_1.txt')[:, 0]
        e = 1000*e  # KeV to eV

    t = np.genfromtxt(f'{prefix}_tscan_1.txt')[:, 0]
    if stage != '3':
        tmp = ['ref_escan_index', 'tscan_index_to_scale']
        for i in tmp:
            if args_dict[i] is None:
                print(f"{i} is required for stage {stage}!")
                return

        ref_escan_index = args.ref_escan_index - 1
        # select tscan for scaling
        tscan_ind_for_scaling = np.array(args.tscan_index_to_scale)
        tscan_energy_for_scale = tscan_energy[tscan_ind_for_scaling-1]
        time_zeros_for_scale = time_zeros[tscan_ind_for_scaling-1]

        # init variable
        if stage != '4':
            escan_data = np.zeros((e.shape[0], num_of_escan))
            escan_data_eps = np.zeros((e.shape[0], num_of_escan))

        tscan_data = np.zeros((t.shape[0], tscan_ind_for_scaling.shape[0]))
        tscan_data_eps = np.zeros((t.shape[0], tscan_ind_for_scaling.shape[0]))

        for i in range(tscan_ind_for_scaling.shape[0]):
            tmp = f'{prefix}_tscan_{tscan_ind_for_scaling[i]}.txt'
            tscan_data[:, i] = np.genfromtxt(tmp)[:, 1]
            tscan_data_eps[:, i] = np.genfromtxt(tmp)[:, 2]

        if stage in ['0', '4']:
            # init scaled_tscan_data and eps
            scaled_tscan_data = np.zeros((t.shape[0], num_of_tscan+1))
            scaled_tscan_data[:, 0] = t
            scaled_tscan_eps = np.zeros((t.shape[0], num_of_tscan))

            if stage == '4':
                e = np.genfromtxt(f'{prefix}_escan_scaled.txt')[:, 0]
                escan_data = \
                    np.genfromtxt(f'{prefix}_escan_scaled.txt')[:, 1:]
                escan_data_eps = \
                    np.genfromtxt(f'{prefix}_escan_eps_scaled.txt')
                A = np.ones(num_of_escan)

            else:
                for i in range(num_of_escan):
                    tmp = f'{prefix}_escan_{i+1}.txt'
                    escan_data[:, i] = np.genfromtxt(tmp)[:, 1]
                    escan_data_eps[:, i] = np.genfromtxt(tmp)[:, 2]
                    tmp = args.parm_A
                    if tmp is None:
                        print(f"parm_A is required for stage {stage}!")
                        return
                    A = np.genfromtxt(args.parm_A)

            scaled_data = automate_scaling(A, ref_escan_index,
                                           e, t,
                                           escan_time, tscan_energy_for_scale,
                                           time_zeros=time_zeros_for_scale,
                                           escan_data=escan_data,
                                           escan_data_eps=escan_data_eps,
                                           tscan_data=tscan_data,
                                           tscan_data_eps=tscan_data_eps,
                                           warn=True)

            A_ref = A[ref_escan_index]*np.ones(escan_time.shape[0])

            np.savetxt(f'{prefix}_A_ref.txt', A_ref)

        else:
            escan_data = \
                np.genfromtxt(f'{prefix}_escan_scaled.txt')[:, 1:]
            escan_data_eps = \
                np.genfromtxt(f'{prefix}_escan_eps_scaled.txt')

            scaled_tscan_data = \
                np.genfromtxt(f'{prefix}_tscan_scaled.txt')
            scaled_tscan_eps = \
                np.genfromtxt(f'{prefix}_tscan_eps_scaled.txt')

            if stage == '1':
                scaled_data = \
                    corr_a_method(ref_escan_index,
                                  e, t,
                                  escan_time,
                                  tscan_energy_for_scale[0],
                                  time_zeros_for_scale[0],
                                  escan_data=escan_data,
                                  escan_data_eps=escan_data_eps,
                                  ref_tscan_data=tscan_data[:, 0],
                                  ref_tscan_data_eps=tscan_data_eps[:, 0])

                escan_data = scaled_data['escan']
                escan_data_eps = scaled_data['escan_eps']

            A = np.ones(num_of_escan)
            scaled_data = automate_scaling(A, ref_escan_index,
                                           e, t,
                                           escan_time, tscan_energy_for_scale,
                                           time_zeros=time_zeros_for_scale,
                                           escan_data=escan_data,
                                           escan_data_eps=escan_data_eps,
                                           tscan_data=tscan_data,
                                           tscan_data_eps=tscan_data_eps,
                                           warn=True)

        scaled_tscan_data[:, tscan_ind_for_scaling] = \
            scaled_data['tscan']
        scaled_tscan_eps[:, tscan_ind_for_scaling - 1] = \
            scaled_data['tscan_eps']

        np.savetxt(f'{prefix}_escan_scaled.txt',
                   np.vstack((e, scaled_data['escan'].T)).T)
        np.savetxt(f'{prefix}_escan_eps_scaled.txt',
                   scaled_data['escan_eps'])
        np.savetxt(f'{prefix}_tscan_scaled.txt', scaled_tscan_data)
        np.savetxt(f'{prefix}_tscan_eps_scaled.txt', scaled_tscan_eps)

    else:
        e = np.genfromtxt(f'{prefix}_escan_scaled.txt')[:, 0]
        escan_data = \
            np.genfromtxt(f'{prefix}_escan_scaled.txt')
        escan_data_eps = \
            np.genfromtxt(f'{prefix}_escan_eps_scaled.txt')
        tscan_data = \
            np.genfromtxt(f'{prefix}_tscan_scaled.txt')
        tscan_data_eps = \
            np.genfromtxt(f'{prefix}_tscan_eps_scaled.txt')

        offset_escan = 2*np.amax(abs(escan_data[:, 1:]))
        offset_tscan = 2*np.amax(abs(tscan_data[:, 1:]))
        escan_time = escan_time[escan_time < np.amax(tscan_data[:, 0])]
        num_of_escan = escan_time.shape[0]
        selected_tscan = np.zeros((num_of_tscan, num_of_escan))
        selected_tscan_eps = np.zeros((num_of_tscan, num_of_escan))

        for i in range(num_of_tscan):
            selected_tscan[i, :] = \
                interp1d(t-time_zeros[i], tscan_data[:, i+1])(escan_time)
            selected_tscan_eps[i, :] = \
                interp1d(t-time_zeros[i], tscan_data_eps[:, i],
                         kind='nearest')(escan_time)

        plt.figure(1)
        for i in range(num_of_escan):
            plt.errorbar(e, escan_data[:, i+1]+i*offset_escan,
                         escan_data_eps[:, i],
                         label=f'{escan_time[i]} ps')
            plt.errorbar(e, i*offset_escan*np.ones(e.shape[0]))
        plt.legend()

        plt.figure(2)
        for i in range(num_of_tscan):
            plt.errorbar(t, tscan_data[:, i+1]+i*offset_tscan,
                         tscan_data_eps[:, i],
                         label=f'{tscan_energy[i]} eV')
            plt.errorbar(t, i*offset_tscan*np.ones(t.shape[0]))
        plt.legend()

        plt.figure(3)
        for i in range(num_of_tscan):
            plt.errorbar(t, tscan_data[:, i+1]+i*offset_tscan,
                         tscan_data_eps[:, i],
                         label=f'{tscan_energy[i]} eV')
            plt.errorbar(t, i*offset_tscan*np.ones(t.shape[0]))
        plt.legend()
        plt.xlim(-1, 2)

        for i in range(num_of_escan):
            plt.figure(4+i)
            plt.errorbar(e, escan_data[:, i+1],
                         escan_data_eps[:, i],
                         label=f'{escan_time[i]} ps')
            plt.errorbar(tscan_energy, selected_tscan[:, i],
                         selected_tscan_eps[:, i],
                         linestyle='none', marker='o', mfc='none')

        plt.show()

    return
