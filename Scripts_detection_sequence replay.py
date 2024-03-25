import mne
from mne.decoding import UnsupervisedSpatialFilter


from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

import sys
sys.path.append('MCCA-main')

from MCCA import MCCA


## Function to select a number of trials to select for each class (equilibrated)
def transform_nb_trials(nb_trials, epochs):
    """
    Input: 
        - nb_trials: number of trials to select for each class
        - epochs: epochs object

    Output:
        - list_trials_to_select: list of boolean to select the trials
    """

    unique_class = list(set(epochs.events[:, 2]))
    list_trials_to_select = []
    dictionary_nb_trials_for_each_class = {unique_class[i]: 0 for i in range(len(unique_class))}
    for i in range(len(epochs.events[:, 2])):
        if dictionary_nb_trials_for_each_class[epochs.events[:, 2][i]] < nb_trials:
            dictionary_nb_trials_for_each_class[epochs.events[:, 2][i]] += 1
            list_trials_to_select.append(True)
        else:
            list_trials_to_select.append(False)
    return list_trials_to_select

## Same function but with events as input
def transform_nb_trials_from_events(nb_trials, events):
    """
    Input:
        - nb_trials: number of trials to select for each class
        - events: list of events

    Output:
        - list_trials_to_select: list of boolean to select the trials
    """ 
    unique_class = list(set(events))
    list_trials_to_select = []
    dictionary_nb_trials_for_each_class = {unique_class[i]: 0 for i in range(len(unique_class))}
    for i in range(len(events)):
        if dictionary_nb_trials_for_each_class[events[i]] < nb_trials:
            dictionary_nb_trials_for_each_class[events[i]] += 1  
            list_trials_to_select.append(True)
        else:
            list_trials_to_select.append(False)
    return list_trials_to_select

## Function to compare the accuracy of a model with bagging and without bagging
def comparison_bagging_or_not(epochs, nb_trials, param_grid):
    """
    Input:
        - epochs: epochs object
        - nb_trials: number of trials to select for each class
    
    Output:
        - best_params: best parameters for the model with bagging
        - no_bagging: accuracy of the model without bagging
        - bagging: accuracy of the model with bagging
    """

    # Select the lines to keep in order to have the same number of trials for each class equal to nb_trials
    list_trials_to_select = transform_nb_trials(nb_trials, epochs)

    bagging = []
    best_params = []

    X_all = epochs.get_data()
    y_all = epochs.events

    for time in range(len(epochs.times)):
        print(time/len(epochs.times)*100)
        
        # Select the data
        X = X_all[list_trials_to_select,:,time]
        y = y_all[list_trials_to_select, 2]

        # Preprocess
        preprocessing = StandardScaler()

        # Choose a model using bagging
        model_bagging = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=10, n_jobs=-2))

        # Hyperparameter search
        search = RandomizedSearchCV(model_bagging, param_grid, n_iter=20, cv=5, n_jobs=-2)
        search.fit(X, y)

        # Cross-validate accuracy scores
        bagging.append(search.best_score_)
        best_params.append(search.best_params_)

    no_bagging = []

    for time in range(len(epochs.times)):
        print(time/len(epochs.times)*100)
        # Select the data
        X = X_all[list_trials_to_select,:,time]
        y = y_all[list_trials_to_select, 2]

        # Preprocess
        preprocessing = StandardScaler()

        # Choose a model
        model_no_bagging = make_pipeline(preprocessing, SVC(random_state=0))

        # Cross-validate accuracy scores
        no_bagging.append(cross_val_score(model_no_bagging, X, y, cv=5).mean())

    return best_params, no_bagging, bagging

## Function to plot the comparison between bagging and no bagging
def plot_comparison(bagging, no_bagging, epochs):
    """
    Input:
        - bagging: accuracy of the model with bagging
        - no_bagging: accuracy of the model without bagging
        - epochs: epochs object
    """

    fig,axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    axes[0].plot(epochs.times, bagging)
    axes[0].hlines(1/6, epochs.times[0], epochs.times[-1], linestyle='dashed', color='black')
    axes[1].plot(epochs.times, no_bagging)
    axes[1].hlines(1/6, epochs.times[0], epochs.times[-1], linestyle='dashed', color='black')
    axes[2].plot(epochs.times, np.array(bagging) - np.array(no_bagging))
    axes[2].hlines(0, epochs.times[0], epochs.times[-1], linestyle='dashed', color='black')
    axes[0].set_title('Accuracy score with bagging')
    axes[1].set_title('Accuracy score without bagging')
    axes[2].set_title('Difference between the two')
    axes[1].set_xlabel('Time (s)')
    axes[0].set_ylabel('Accuracy score')
    fig.suptitle('Comparison between bagging and no bagging')
    plt.tight_layout()
    plt.show()

## Function to test the strategy to take timeframes instead of points
def time_frame(epochs, nb_time_points, nb_trials_per_class, param_grid, meaning_over_time = False):
    """
    Input:
        - epochs: epochs object
        - nb_time_points: number of time points to take each side of the current time point
        - nb_trials_per_class: number of trials to select for each class
        - param_grid: dictionary of hyperparameters
        - meaning_over_time: if True, the mean of the time frame is taken

    Output:
        - results: list of accuracy scores
    """

    # Select the lines to keep in order to have the same number of trials for each class equal to nb_trials
    list_trials_to_select = transform_nb_trials(nb_trials_per_class, epochs)

    results = []

    X_all = epochs.get_data()[list_trials_to_select, :, :]
    y_all = epochs.events[list_trials_to_select, 2]

    for time in range(nb_time_points, len(epochs.times) - nb_time_points):
        print(time/len(epochs.times)*100)

        # Select the data
        if meaning_over_time:
            X = np.mean(X_all[:,:,time-nb_time_points:time+nb_time_points+1], axis=2)
        else:
            X = X_all[:,:,time-nb_time_points:time+nb_time_points+1].reshape(X_all.shape[0], -1)
        y = y_all

        # Preprocess
        preprocessing = StandardScaler()

        # Choose a model
        model = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=5, n_jobs=-2))

        # Hyperparameter search
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, n_jobs=-2)
        search.fit(X, y)

        # Cross-validate accuracy scores
        results.append(search.best_score_)

    return results

## Function to plot the results of the time frame strategy
def plot_time_frames(list_results, epochs, list_nb_time_points, meaning_over_time = False):
    """
    Input:
        - list_results: list of accuracy scores
        - epochs: epochs object
        - list_nb_time_points: list of number of time points to take each side of the current time point
        - meaning_over_time: if True, the mean of the time frame is taken
    """

    colorbar = plt.cm.get_cmap('viridis', len(list_results))
    
    for results, color, nb_time_points in zip(list_results, colorbar.colors, list_nb_time_points):
        if nb_time_points == 0:
            plt.plot(epochs.times, results, color=color, label=f'number of time points = {nb_time_points+1}')
        else:
            plt.plot(epochs.times[nb_time_points:-nb_time_points], results, color=color, label=f'number of time points = {nb_time_points*2+1}')
    plt.hlines(1/6, epochs.times[nb_time_points], epochs.times[-nb_time_points], linestyle='dashed', color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy score')
    plt.legend()
    if meaning_over_time:
        plt.title('Accuracy score with meaning over time')
    else:
        plt.title('Accuracy score without meaning over time')
    plt.show()

## Function to search for the best hyperparameters
def search_for_best_hyperparameters(epochs, nb_trials, param_grid, best_time):
    """
    Input:
        - epochs: epochs object
        - nb_trials: number of trials to select for each class
        - param_grid: dictionary of hyperparameters
        - best_time: time point of maximum accuracy score
    
    Output:
        - best_params: best parameters for the model
        - all_results: all the results of the search
    """

    X_all = epochs.get_data()
    y_all = epochs.events

    best_results = []
    all_results = []
    best_params = {key: [] for key in param_grid.keys()}

    #We split the data in 5 parts and we do the search on each part

    X_all_split = np.array_split(X_all, 5)
    y_all_split = np.array_split(y_all, 5)

    for i in range(len(X_all_split)):
        print(i/len(X_all_split))
        list_trials_to_select = transform_nb_trials_from_events(nb_trials, y_all_split[i][:, 2])

        # Select the data
        X = X_all_split[i][list_trials_to_select,:,best_time]
        y = y_all_split[i][list_trials_to_select, 2]

        # Preprocess
        preprocessing = StandardScaler()

        # Choose a model using bagging
        model_bagging = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=20, n_jobs=-3))

        # Hyperparameter search
        search = RandomizedSearchCV(model_bagging, param_grid, n_iter=300, cv=5, n_jobs=-2)
        search.fit(X, y)

        # Cross-validate accuracy scores
        best_results.append(search.best_score_)
        all_results.append(search.cv_results_)
        for key in param_grid.keys():
            best_params[key].append(search.best_params_[key])

    return best_params, all_results

## Function to plot the results of the search for the best hyperparameters
def plot_best_hyperparameters(best_params, all_results, param_grid):
    """
    Input:
        - best_params: best parameters for the model
        - all_results: all the results of the search
        - param_grid: dictionary of hyperparameters
    """

    # Plot the best parameters at best time
    fig, axes = plt.subplots(len(param_grid.keys()), 1, figsize=(10, 20))
    for i, key in enumerate(param_grid.keys()):
        axes[i].plot(best_params[key])
        axes[i].set_title(key)
        if (type(param_grid[key][0]) == int or type(param_grid[key][0]) == float):
            axes[i].set_yscale('log')
    plt.tight_layout()

    # Plot the statistics of each hyperparameter at best time
    dfs = []
    for i in range(len(all_results)):
        dfs.append(pd.DataFrame(all_results[i]))
    df = pd.concat(dfs)

    fig, axes = plt.subplots(len(param_grid.keys()), 1, figsize=(10, 20))

    for i, key in enumerate(param_grid.keys()):
        to_boxplot = []
        for value in param_grid[key]:
            to_boxplot.append(df.loc[df["param_" + key] == value, "mean_test_score"].to_list())
        axes[i].boxplot(to_boxplot, labels=param_grid[key])
        axes[i].set_title(key)
    plt.tight_layout()

## Function to test the effect of PCA
def effects_of_pca(epochs, param_grid, best_time):
    """
    Input:
        - epochs: epochs object
        - param_grid: dictionary of hyperparameters
        - best_time: time point of maximum accuracy score

    Output:
        - results_no_PCA: list of accuracy scores without PCA
        - results_PCA: list of accuracy scores with PCA
        - number_of_components: list of number of components
    """

    X_all = epochs.get_data()
    y_all = epochs.events

    pca = UnsupervisedSpatialFilter(PCA(), average=False)
    pca_data = pca.fit_transform(X_all)

    results_no_PCA = []

    number_of_components = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for i in range(10):
        print(i/10*50)
        results_no_PCA.append([])
        for number_of_component in number_of_components:
            list_trials_to_select = transform_nb_trials(number_of_component, epochs)

            # Select the data
            X = X_all[list_trials_to_select,:,36]
            y = y_all[list_trials_to_select, 2]

            # Preprocess
            preprocessing = StandardScaler()

            # Choose a model using bagging
            model_bagging = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=5, n_jobs=-3))

            # Hyperparameter search
            search = RandomizedSearchCV(model_bagging, param_grid, n_iter=20, cv=5, n_jobs=-2)
            search.fit(X, y)

            # Cross-validate accuracy scores
            results_no_PCA[-1].append(search.best_score_)

    results_PCA = []
    
    for i in range(10):
        print(i/10*50 + 50)
        results_PCA.append([])
        for number_of_component in number_of_components:
            list_trials_to_select = transform_nb_trials(number_of_component, epochs)

            # Select the data
            X = pca_data[list_trials_to_select,:,best_time]
            y = y_all[list_trials_to_select, 2]

            # Preprocess
            preprocessing = StandardScaler()

            # Choose a model using bagging
            model_bagging = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=5, n_jobs=-3))

            # Hyperparameter search
            search = RandomizedSearchCV(model_bagging, param_grid, n_iter=20, cv=5, n_jobs=-2)
            search.fit(X, y)

            # Cross-validate accuracy scores
            results_PCA[-1].append(search.best_score_)

    return results_no_PCA, results_PCA, number_of_components

## Function to plot the effects of PCA
def plot_effects_of_pca(results_no_PCA, results_PCA, number_of_components):
    """
    Input:
        - results_no_PCA: list of accuracy scores without PCA
        - results_PCA: list of accuracy scores with PCA
        - number_of_components: list of number of components
    """

    for i in range(len(results_no_PCA)):
        plt.plot(number_of_components, results_no_PCA[i], label='without PCA', alpha=0.2, color='orange')
        plt.plot(number_of_components, results_PCA[i], label='with PCA', alpha=0.2, color = 'blue')

    line1 = plt.plot(number_of_components, np.mean(results_no_PCA, axis=0), label='without PCA', color='orange')
    line2 = plt.plot(number_of_components, np.mean(results_PCA, axis=0), label='with PCA', color='blue')
    plt.xlabel('Number of components')
    plt.ylabel('Accuracy score')
    plt.legend(handles=[line1[0], line2[0]])
    plt.show()

## Function that implements the micro-averaging strategy
def micro_averaging(epochs_data, epochs_events, nb_trials, nb_trial_per_average):
    """
    Input:
        - epochs_data: data of the epochs
        - epochs_events: events of the epochs
        - nb_trials: number of trials to select for each class
        - nb_trial_per_average: number of trials to average to make a new trial
    
    Output:
        - new_epochs: new epochs
        - events: new events
    """

    # List of the unique classes (6 here)
    unique_class = list(set(epochs_events))

    # Extracting the position of the trials for each class to stay balanced
    dictionary_position_trials_for_each_class = {unique_class[i]: [] for i in range(len(unique_class))}
    for i in range(len(epochs_events)):
        if len(dictionary_position_trials_for_each_class[epochs_events[i]]) < nb_trials * nb_trial_per_average:
            for j in range(nb_trial_per_average):
                dictionary_position_trials_for_each_class[epochs_events[i]].append(i)

    # Creating the new epochs by drawing randomly the trials and averaging them
    new_epochs = []
    events = []
    for i in range(len(unique_class)):
        for _ in range(nb_trials):
            draw_index = np.arange(0,len(dictionary_position_trials_for_each_class[unique_class[i]]))
            np.random.shuffle(draw_index)
            draw_index = draw_index[:nb_trial_per_average]
            draw = [dictionary_position_trials_for_each_class[unique_class[i]].pop(index) for index in sorted(draw_index,reverse=True)]
            new_epochs.append(np.mean(epochs_data[draw], axis=0))
            events.append(unique_class[i])
    new_epochs = np.array(new_epochs)

    return new_epochs, events

def machine_learning_for_micro_averaging(epochs, nb_trials, nb_trial_per_average, param_grid):
    """
    Input:
        - epochs: epochs object
        - nb_trials: number of trials to select for each class
        - nb_trial_per_average: number of trials to average to make a new trial
        - param_grid: dictionary of hyperparameters
    
    Output:
        - results: list of accuracy scores
    """

    epochs_train, epochs_test, events_train, events_test = train_test_split(epochs.get_data(), epochs.events[:,2], test_size=0.2, random_state=0)

    micro_averaging_epochs_train, events_train = micro_averaging(epochs_train, events_train, int(nb_trials*0.8), nb_trial_per_average)

    results = []

    for time in range(len(epochs.times)):
        print(time/len(epochs.times)*100)
        # Select the data
        X = micro_averaging_epochs_train[:,:,time]
        y = events_train

        # Preprocess
        preprocessing = StandardScaler()

        # Choose a model using bagging
        model_bagging = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=10, n_jobs=-2))

        # Hyperparameter search
        search = RandomizedSearchCV(model_bagging, param_grid, n_iter=10, cv=5, n_jobs=-2)
        search.fit(X, y)

        # Cross-validate accuracy scores on the test set
        X = epochs_test[:,:,time]
        y = events_test
        results.append(search.score(X, y))

    return results

## Function to plot the results of the micro-averaging strategy
def plot_microaveraging(baggings, epochs):
    """
    Input:
        - baggings: list of accuracy scores
        - epochs: epochs object
    """

    fig,axes = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(baggings)))
    labels = ['1', '5', '10']
    for i,bagging in enumerate(baggings):
        axes.plot(epochs.times, bagging, color = colors[i], label = labels[i])
    axes.hlines(1/6, epochs.times[0], epochs.times[-1], linestyle='dashed', color="black")
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Accuracy score')
    axes.legend()
    fig.suptitle('Effect of micro averaging')
    plt.tight_layout()
    plt.show()

## Function to test the effect of the number of trials
def test_nb_trials(epochs, list_nb_trials, param_grid):
    """
    Input:
        - epochs: epochs object
        - list_nb_trials: list of number of trials to select for each class
        - param_grid: dictionary of hyperparameters
    
    Output:
        - results_to_plot: list of accuracy scores
    """

    results_to_plot = []

    X_all = epochs.get_data()
    y_all = epochs.events

    for nb_trials in list_nb_trials:
        # Select the lines to keep in order to have the same number of trials for each class equal to nb_trials
        list_trials_to_select = transform_nb_trials(nb_trials, epochs)

        # Define the parameter grid

        results = []

        for i in range(5):
            results.append([])
            
            for time in range(len(epochs.times)):
                print(nb_trials, time/len(epochs.times)*100)

                # Select the data
                X = X_all[list_trials_to_select,:,time]
                y = y_all[list_trials_to_select, 2]

                # Preprocess
                preprocessing = StandardScaler()

                # Choose a model using bagging
                model_bagging = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=10, n_jobs=-2))

                # Hyperparameter search
                search = RandomizedSearchCV(model_bagging, param_grid, n_iter=20, cv=5, n_jobs=-2)
                search.fit(X, y)

                # Cross-validate accuracy scores
                results[-1].append(search.best_score_)

        results_to_plot.append(results)
    
    return results_to_plot

## Function to plot the results of the effect of the number of trials with accuracy over time
def plot_nb_trials(epochs, results_to_plot, list_nb_trials):
    fig,ax = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
    viridis = mpl.colormaps['viridis']
    real_times = epochs.times
    for i in range(len(results_to_plot)):
        for j in range(len(results_to_plot[i])):
            ax.plot(real_times, results_to_plot[i][j], color = viridis(i/(len(results_to_plot)-1)), alpha=0.2)
        ax.plot(real_times, np.mean(results_to_plot[i], axis=0), color = viridis(i/len(results_to_plot)))
    ax.hlines(1/6, real_times[0], real_times[-1], linestyle='dashed', color='black')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=list_nb_trials[0], vmax=list_nb_trials[-1]), cmap=viridis), ax=ax, label='Number of samples')
    fig.suptitle('Accuracy score with bagging for different number of samples')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accuracy score')
    plt.show() 

## Function to test the effect of the number of trials with maximum accuracy
def plot_maximum_accuracy(results_to_plot, list_nb_trials):
    abscissa = list_nb_trials
    ordinate = np.max(np.array(results_to_plot), axis=2).T

    plt.boxplot(ordinate)
    plt.xlabel('Number of samples')
    plt.ylabel('Maximum ccuracy score')
    plt.xticks(np.arange(1, len(abscissa)+1), abscissa)
    plt.show()

def intra_subject_decoder(epochs, param_grid, n_folds=5):
    """
    Splits data from each subject into 80% training and 20% testing. MCCA is applied
    to averaged training data from all subjects, and the weights are used to transform
    all single-trial data into MCCA space. Intra-subject decoders are trained on
    single-trial data in MCCA space for each subject and the results are saved to file.

    Parameters:
        n_folds (int): Number of cross-validation folds

    """
    X, y, train, test, data_averaged = _load_data(epochs, n_folds=n_folds)
    result_MCCA_t = []
    result_no_MCCA_t = []
    for t in range(len(epochs.times)):
        print('Time point', t, 'of', len(epochs.times))
        y_true_all = []
        y_pred_all = []
        y_pred_no_mcca_all = []
        BAs = []
        BAs_no_mcca = []
        n_subjects = len(y)
        for i in range(n_subjects):
            for j in range(n_folds):
                mcca = MCCA(50,10,0)
                mcca.obtain_mcca(data_averaged[j])
                X_mcca = mcca.transform_trials(X[i], subject=i) # .reshape((len(y[i]), -1))
                X_mcca = X_mcca[:,t,:].reshape((len(y[i]), -1))
                X_no_mcca = X[i][:,t,:].reshape((len(y[i]), -1))
                X_train = X_mcca[train[j][i]]
                y_train = y[i][train[j][i]]
                X_test = X_mcca[test[j][i]]
                y_test = y[i][test[j][i]]
                X_no_mcca_train = X_no_mcca[train[j][i]]
                X_no_mcca_test = X_no_mcca[test[j][i]]
                preprocessing = StandardScaler()
                model_bagging = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=10, n_jobs=-2))
                clf = RandomizedSearchCV(model_bagging, param_grid, n_iter=20, cv=5, n_jobs=-2)
                clf.fit(X_train, y_train)
                preprocessing = StandardScaler()
                model_bagging = make_pipeline(preprocessing, BaggingClassifier(SVC(random_state=0), n_estimators=10, n_jobs=-2))
                clf_no_mcca = RandomizedSearchCV(model_bagging, param_grid, n_iter=20, cv=5, n_jobs=-2)
                clf_no_mcca.fit(X_no_mcca_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_no_mcca = clf_no_mcca.predict(X_no_mcca_test)
                y_true_all.append(y_test)
                y_pred_all.append(y_pred)
                y_pred_no_mcca_all.append(y_pred_no_mcca)
                BAs.append(balanced_accuracy_score(y_test, y_pred))
                BAs_no_mcca.append(balanced_accuracy_score(y_test, y_pred_no_mcca))
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_pred_no_mcca_all = np.concatenate(y_pred_no_mcca_all)
        result_MCCA_t.append(BAs)
        result_no_MCCA_t.append(BAs_no_mcca)
    return result_MCCA_t, result_no_MCCA_t

def _load_data(epochs, n_folds=5):
    """
    Splits data from each subject into 80% training and 20% testing. MCCA is applied
    to averaged training data from all subjects, and the weights are used to transform
    all single-trial data into MCCA space.

    Parameters:
        n_folds (int): Number of cross-validation folds

    Returns:
        X (ndarray): The transformed data in MCCA space, flattened across MCCA
            and time dimensions for the classifier.
            (subjects x trials x (samples x MCCAs))

        y (ndarray): Labels corresponding to trials in X. (subjects x trials)

        train_indices (list): Indices of trials used in the training set for
            each subject.

        test_indices (list): Indices of trials used in the test set for each
            subject.

        data_averaged (ndarray): Averaged data for each subject in MCCA space.
    """
    X = []
    y = []
    epo = epochs.copy()
    train_indices = [[] for _ in range(n_folds)]
    test_indices = [[] for _ in range(n_folds)]
    data_averaged = [[] for _ in range(n_folds)]
    
    # We split the data as if we had two subjects
    n_epochs = len(epochs)
    y_label = epo.events[:, -1]
    X_epo = epo.get_data()
    X_ = np.array([X_epo[:int(n_epochs/2)], X_epo[int(n_epochs/2):]])
    X_ = np.swapaxes(X_, 2, 3)
    y_ = np.array([y_label[:int(n_epochs/2)], y_label[int(n_epochs/2):]])

    n_subjects = len(y_)
    for i in range(n_subjects):
        data_st, labels = X_[i], y_[i]
        X.append(data_st)
        y.append(labels)
        sss = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        j = 0
        for train, test in sss.split(data_st, labels):
            train_indices[j].append(train)
            test_indices[j].append(test)
            evoked1 = np.mean(data_st[train][np.where(labels[train] == 1)], axis=0)
            evoked2 = np.mean(data_st[train][np.where(labels[train] == 2)], axis=0)
            evoked3 = np.mean(data_st[train][np.where(labels[train] == 3)], axis=0)
            evoked4 = np.mean(data_st[train][np.where(labels[train] == 4)], axis=0)
            evoked5 = np.mean(data_st[train][np.where(labels[train] == 5)], axis=0)
            evoked6 = np.mean(data_st[train][np.where(labels[train] == 6)], axis=0)
            evoked = [evoked1, evoked2, evoked3, evoked4, evoked5, evoked6]
            data_averaged[j].append(np.concatenate(evoked, axis=0))
            j += 1

    for j in range(n_folds):
        data_averaged[j] = np.stack(data_averaged[j], axis=0)

    return X, y, train_indices, test_indices, data_averaged

# Function to plot the results of the intra-subject decoder
def plot_result_microaveraging(epochs, result_MCCA_t, result_no_MCCA_t):
    """
    Input:
        - epochs: epochs object
        - result_MCCA_t: list of accuracy scores with MCCA
        - result_no_MCCA_t: list of accuracy scores without MCCA
    """

    mean_result_MCCA_t = np.mean(result_MCCA_t, axis=1)
    mean_result_no_MCCA_t = np.mean(result_no_MCCA_t, axis=1)
    std_result_MCCA_t = np.std(result_MCCA_t, axis=1)
    std_result_no_MCCA_t = np.std(result_no_MCCA_t, axis=1)
    plt.plot(epochs.times, mean_result_MCCA_t, label='MCCA')
    plt.fill_between(epochs.times, mean_result_MCCA_t - std_result_MCCA_t, mean_result_MCCA_t + std_result_MCCA_t, alpha=0.2)
    plt.plot(epochs.times, mean_result_no_MCCA_t, label='No MCCA')
    plt.fill_between(epochs.times, mean_result_no_MCCA_t - std_result_no_MCCA_t, mean_result_no_MCCA_t + std_result_no_MCCA_t, alpha=0.2)
    plt.hlines(1/6, epochs.times[0], epochs.times[-1], label='Chance', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Balanced accuracy')
    plt.legend()
    plt.show()
