import pandas as pd
import seaborn as sns
from ._helpers.stats import *
from .brain import Brain
import seaborn as sns
import nibabel as nib

class Model(object):
    """
    superEEG model and associated locations

    This class holds your superEEG model.  To create an instance, you need a model
    (an electrode x electrode correlation matrix), electrode locations in MNI
    space, and the number of subjects used to create the model.  Additionally,
    you can include a meta dictionary with any other information that you want
    to save with the model.

    Parameters
    ----------

    data : list
        A list of superEEG.Brain objects used to create the model

    locs : pandas.DataFrame
        MNI coordinate (x,y,z) by number of electrode df containing electrode locations

    template : filepath
        Path to a template nifti file used to set model locations

    meta : dict
        Optional dict containing whatever you want

    Attributes
    ----------

    numerator : Numpy.ndarray
        A locations x locations matrix comprising the sum of the zscored
        correlation matrices over subjects

    denominator : Numpy.ndarray
        A locations x locations matrix comprising the sum of the number of
        subjects contributing to each matrix cell

    n_subs : int
        Number of subject used to create the model


    Returns
    ----------
    model : superEEG.Model instance
        A model that can be used to infer timeseries from unknown locations

    """

    def __init__(self, data=None, locs=None, template='../superEEG/data/MNI152_T1_6mm_brain.nii.gz',
                 measure='kurtosis', threshold=10, meta={}):

        # get locations from template, or from locs arg
        if locs is None:

            # load in template file
            img = nib.load(template)

            # get locations from template
            self.locs = pd.DataFrame(nii2cmu(img), columns=['x', 'y', 'z'])

        else:

            # otherwise, create df from locs passed as arg
            self.locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

        # initialize numerator
        numerator = np.zeros((self.locs.shape[0], self.locs.shape[0]))

        # initialize denominator
        denominator = np.zeros((self.locs.shape[0], self.locs.shape[0]))

        # loop over brain objects
        for bo in data:

            # filter bad electrodes
            bo = filter_elecs(bo, measure=measure, threshold=threshold)

            # get subject-specific correlation matrix
            sub_corrmat = r2z(get_corrmat(bo))

            # get rbf weights
            sub_rbf_weights = rbf(self.locs, bo.locs)

            #  get subject expanded correlation matrix
            num_corrmat_x, denom_corrmat_x = get_expanded_corrmat_lucy(sub_corrmat, sub_rbf_weights)

            # set weights equal to zero where the numerator is equal to nan
            denom_corrmat_x[np.isnan(num_corrmat_x)] = 0

            # add in new subj data to numerator
            numerator = np.nansum(np.dstack((numerator, num_corrmat_x)), 2)

            # add in new subj data to denominator
            denominator += denom_corrmat_x

        # attach numerator
        self.numerator = numerator

        # attach denominator
        self.denominator = denominator

        # attach number of subjects
        self.n_subs = len(data)

        # meta
        self.meta = meta

    def predict(self, bo, tf=False, kthreshold=10):
        """
        Takes a brain object and a 'full' covariance model, fills in all
        electrode timeseries for all missing locations and returns the new brain object

        Parameters
        ----------

        bo : Brain data object or a list of Brain objects
            The brain data object that you want to predict

        tf : bool
            If True, uses Tensorflow (default is False).

        Returns
        ----------

        bo_p : Brain data object
            New brain data object with missing electrode locations filled in

        """

        # filter bad electrodes
        bo = filter_elecs(bo, measure='kurtosis', threshold=kthreshold)

        # get subject-specific correlation matrix
        sub_corrmat = r2z(get_corrmat(bo))

        # get rbf weights
        sub_rbf_weights = rbf(self.locs, bo.locs)

        #  get subject expanded correlation matrix
        num_corrmat_x, denom_corrmat_x = get_expanded_corrmat_lucy(sub_corrmat, sub_rbf_weights)

        # set weights equal to zero where the numerator is equal to nan
        denom_corrmat_x[np.isnan(num_corrmat_x)] = 0

        # add in new subj data
        model_corrmat_x = np.divide(np.nansum(np.dstack((self.numerator.as_matrix(), num_corrmat_x)), 2), self.denominator + denom_corrmat_x)

        # replace the diagonal with zeros
        model_corrmat_x[np.eye(model_corrmat_x.shape[0]) == 1] = 0

        # convert nans to zeros
        model_corrmat_x[np.where(np.isnan(model_corrmat_x))] = 0

        # expanded rbf weights
        model_rbf_weights = rbf(pd.concat([self.locs, bo.locs]), self.locs)

        # get model expanded correlation matrix
        num_corrmat_x, denom_corrmat_x = get_expanded_corrmat_lucy(model_corrmat_x, model_rbf_weights)

        # divide the numerator and denominator
        model_corrmat_x = np.divide(num_corrmat_x, denom_corrmat_x)

        #convert from z to r
        model_corrmat_x = z2r(model_corrmat_x)

        # convert diagonals to 1
        model_corrmat_x[np.where(np.isnan(model_corrmat_x))] = 1

        # timeseries reconstruction
        if tf:
            reconstructed = reconstruct_activity_tf(bo, model_corrmat_x)
        else:
            reconstructed = reconstruct_activity(bo, model_corrmat_x)

        # # create new bo with inferred activity
        return Brain(data=reconstructed, locs=pd.concat([self.locs, bo.locs]),
                    sessions=bo.sessions, sample_rate=bo.sample_rate)

    def expand(self, template):
        """
        Expand a model to a template space

        Parameters
        ----------

        template : Model object

        Returns
        ----------

        new_model : Model object
            New model object in template space

        """

        # expanded rbf weights
        model_rbf_weights = rbf(pd.concat([self.locs, template.locs]), template.locs)

        # return new model
        return Model(data=get_expanded_corrmat(self.data.as_matrix(), model_rbf_weights),
                     locs=pd.concat([self.locs, template.locs]))

    def update(self, bo):
        """
        Update a model with new data

        Parameters
        ----------

        bo : Brain object
            New subject data

        Returns
        ----------

        new_model : Model object
            New model object updated with new subject data

        """

        # get subject-specific correlation matrix
        sub_corrmat = get_corrmat(bo)

        # if the locations are the same, skip the expand steps
        if self.locs==bo.locs:

            model_corrmat_x = sub_corrmat

        else:

            # get rbf weights
            sub_rbf_weights = rbf(pd.concat([self.locs, bo.locs]), bo.locs)

            #  get subject expanded correlation matrix
            sub_corrmat_x = get_expanded_corrmat(sub_corrmat, sub_rbf_weights)

            # expanded rbf weights
            model_rbf_weights = rbf(pd.concat([self.locs, bo.locs]), self.locs)

            # get model expanded correlation matrix
            model_corrmat_x = get_expanded_corrmat(self.data.as_matrix(), model_rbf_weights)

        # add in new subj data
        model_corrmat_x = np.divide(((model_corrmat_x * self.n_subs) + sub_corrmat_x), (self.n_subs+1))

        #convert from z to r
        model_corrmat_x = z2r(model_corrmat_x)

        # return a new updated model
        return Model(data=model_corrmat_x, locs=pd.concat([self.locs, bo.locs]),
                     n_subs=self.n_subs+1, meta=self.meta)

    def plot(self):
        """
        Plot the superEEG model
        """
        sns.heatmap(self.data, xticklabels=False, yticklabels=False)
        sns.plt.title('SuperEEG Model, N=' + str(self.n_subs))
        sns.plt.show()
