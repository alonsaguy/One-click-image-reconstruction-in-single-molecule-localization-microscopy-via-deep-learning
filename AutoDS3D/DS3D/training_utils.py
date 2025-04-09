import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from torch.utils.data import DataLoader, Dataset
from typing import List, NamedTuple
import pickle


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    batch_accuracy: float


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]



class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """
    def __init__(self, model, loss_fn, optimizer, lr_scheduler=None, device=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param lr_scheduler: learning rate adjustment if given lr_scheduler
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        if self.device:
            model.to(self.device)  # put the model into the designated device

    # ********* change this part accordingly
    def fit(self,
            dl_train: DataLoader,
            dl_test: DataLoader,
            num_epochs,
            checkpoints: dict = None,  # the path and name of the net to be saved in training
            early_stopping: int = None,  # stop training if there is no improvement for this number of epochs
            print_every=1,  # print period (epoch), the first and last epoch are mandatory
            post_epoch_fn: Callable = None,  # what to do after each epoch
            **kw,  # other parameters
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.

        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0  # indicator of current epoch
        train_loss, train_acc, test_loss, test_acc = [], [], [], []  # lists for epoch results

        best_metric = None  # a metric for raising: model save, early stopping and learning rate adjustment
        epochs_without_improvement = 0  # the number of epochs that best_loss is not updated
        for epoch in range(num_epochs):
            # print sth at the beginning of one epoch
            verbose = False
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---", verbose)

            # training for one epoch
            epoch_train_result = self.train_epoch(dl_train, verbose=verbose, **kw)  # get a EpochResult
            train_loss.append(sum(epoch_train_result.losses)/len(epoch_train_result.losses))
            train_acc.append(epoch_train_result.accuracy)
            epoch_test_result = self.test_epoch(dl_test, verbose=verbose, **kw)  # get a EpochResult
            test_loss.append(sum(epoch_test_result.losses)/len(epoch_test_result.losses))
            test_acc.append(epoch_test_result.accuracy)

            # what to do after one epoch of training and test
            actual_num_epochs += 1
            # current_average_metric = epoch_test_result.accuracy  # the last average loss. test_loss is the list of epoch losses
            current_average_metric = test_loss[-1]
            # early stopping? save the model?
            if best_metric is not None and current_average_metric > best_metric:  # if accuracy is getting worse
                epochs_without_improvement += 1
            else:  # if test accuracy is improved
                best_metric = current_average_metric
                epochs_without_improvement = 0

                if checkpoints is not None:  # if checkpoints is given, save the net
                    checkpoints['state_dict'] = self.model.state_dict()
                    torch.save(checkpoints, checkpoints['file_name'])
                    print(f"\n*** Saved checkpoint {checkpoints['file_name']}")

            if early_stopping and epochs_without_improvement == early_stopping:
                break

            # update learning rate?
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(current_average_metric)

            # in case sth should be done after each epoch
            if post_epoch_fn:
                pass
                # post_epoch_fn(epoch, train_results, test_results, verbose)

            # save loss curves at the end of each epoch
            # if checkpoints is not None:
            #     loss_curves = dict(train_loss=train_loss, test_loss=test_loss)
            #     with open(f'{checkpoints}.pkl', 'wb') as handle:
            #         pickle.dump(loss_curves, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # ========================

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)  # return a FitResult

    def save_checkpoint(self, checkpoints: dict):
        """
        Saves the current model to a file with the given name (treated as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        # saved_state = torch.load(f'{checkpoint_file}.pt', map_location=device)
        # net.load_state_dict(saved_state['model_state'])
        # dirname = os.path.dirname(checkpoint_filename)
        # dirname = checkpoints['path']
        # os.makedirs(dirname, exist_ok=True)  # if the dir exists, it is ok.
        # checkpoints['state_dict'] = self.model.state_dict()
        # checkpoints['optimizer'] = self.optimizer.state_dict()
        # checkpoints['train_loss'] = self.optimizer.state_dict()
        # checkpoints['test_loss'] = self.optimizer.state_dict()
        #
        # torch.save(self.model.state_dict(), dirname)
        # # save other things??
        #
        # print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).

        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # from nn.Module. To do so, some layers, like dropout, know of their mode switching
        return self._foreach_batch(dl_train, self.train_batch, kw['verbose'])

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, kw['verbose'])

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.

        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        pass

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.

        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        pass

    @staticmethod  # notice, there is no self in this method, can be used without instantiation
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    # ********* change this part accordingly
    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],  # for each batch
            verbose=True,
            max_batches=None,
    ) -> EpochResult:
        """
        one epoch
        Evaluates the given forward-function on batches from the given dataloader, and prints progress along the way.
        """
        batch_losses = []  # a list to save each batch loss
        batch_accuracy = []  # for classification task
        # num_samples = len(dl.sampler)  # torch.utils.data.sampler.RandomSampler, how many samples, be careful
        num_batches = len(dl.batch_sampler)  # torch.utils.data.sampler.BatchSampler, how many batches, be careful

        # if max_batches is not None:  # if there is maximum batches. in case you don't want to use all the batches
        #     if max_batches < num_batches:
        #         num_batches = max_batches
        #         num_samples = num_batches * dl.batch_size

        # combine sys.stdout, os.devnull and tpdm to show *progress bar* in one epoch
        if verbose:
            pbar_file = sys.stdout  # similar to print, but have no '/n' at the end, then pbar_file.write() is used
        else:
            pbar_file = open(os.devnull, "w")
        pbar_name = forward_fn.__name__  # set the name of the progress bar as the batch training name
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):

                data = next(dl_iter)  # get one batch of data, x and y
                batch_res = forward_fn(data)  # batch training, get one a batch result

                # pbar.set_description(f"{pbar_name} (Loss {batch_res.loss:.4f} | J_idx {batch_res.batch_accuracy:.4f})")  # give sth to show and update in progress
                pbar.set_description(f"{pbar_name} (Loss {batch_res.loss:.4f})")
                pbar.update()  # update the expected time

                # record, at the end of one batch
                batch_losses.append(batch_res.loss)    # BatchResult
                batch_accuracy.append(batch_res.batch_accuracy)

            avg_loss = sum(batch_losses) / num_batches  # average loss of this epoch
            avg_accuracy = sum(batch_accuracy) / num_batches  #

            # after one epoch, show sth (before the progress bar)
            # pbar.set_description(f"(Avg. Loss {avg_loss:.4f} | Avg. J_idx {avg_accuracy:.04f})")
            pbar.set_description(f"(Avg. Loss {avg_loss:.4f})")
        return EpochResult(losses=batch_losses, accuracy=avg_accuracy)




class TorchTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, lr_scheduler=None, device=None):
        super().__init__(model, loss_fn, optimizer, lr_scheduler, device)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch  # unpacking
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()

        jacc_ind = 0

        return BatchResult(loss=loss.item(), batch_accuracy=jacc_ind)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            output = self.model(X)
            loss = self.loss_fn(output, y)

            jacc_ind = 0

        return BatchResult(loss=loss.item(), batch_accuracy=jacc_ind)



