# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt

OUT_DIR = 'diff_evolution'  # output directory for logs
EXP_ID = 'default'  # the ID of this experiment (used to create log names)


# read the summary log and plot the experiment
# evals, lower, mean, upper = utils.get_plot_data(
#     OUT_DIR, EXP_ID + '.' + 'f01')
plt.figure(figsize=(12, 8))
# utils.plot_experiment(evals, lower, mean, upper,
#                       legend_name='Default settings')

fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']
#fit_names = ['f01']
#experiments = ['default', 'adapt_variance', 'adapt_fitness_probability']

# for f_name in fit_names:
#     exp_ids = map(lambda x: x + '.' + f_name, experiments)
#     plt.xscale('log')
#     utils.plot_experiments(OUT_DIR, exp_ids, stat_type='objective')
#     plt.savefig(f_name + 'fitness.png')
#     plt.cla()


fit_name = 'f10'
stat_type = "fitness"
plt.yscale('log')
# plt.xscale('log')
# utils.plot_experiments(
#     OUT_DIR, ['default' + '.' + fit_name, 'adapt_variance' + '.' + fit_name, 'adapt_fitness_probability.' + fit_name], stat_type=stat_type)
# plt.savefig(fit_name + stat_type + ".png")

experiments = ['diff_evolution' + '.' +
               name for name in fit_names]
# utils.plot_experiments(
#     OUT_DIR, experiments, stat_type=stat_type)
# plt.savefig("diff_evolution_" + stat_type + ".png")
# utils.plot_experiments(
#     OUT_DIR, ['diff_evolution' + '.' + "f01"], stat_type=stat_type)
# plt.savefig("diff_evolution_" + stat_type + ".png")


# for name in fit_names:
#     plt.xscale('log')

#     utils.plot_experiments(
#         OUT_DIR, ['diff_evolution' + '.' + name], stat_type=stat_type)
#     plt.savefig("diff_evolution_" + name + stat_type + ".png")
#     plt.close()
for name in fit_names:
    utils.plot_experiments(
        OUT_DIR, ['diff_evolution' + '.' + name], stat_type=stat_type)
plt.savefig("diff_evolution_" + stat_type + ".png")
