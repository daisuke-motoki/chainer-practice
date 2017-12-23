from chainer import datasets, serializers, iterators, optimizers, training
from chainer.training import extensions
from nets import CapsNet


if __name__ == "__main__":
    result_dir = "result/mnist"
    train, test = datasets.get_mnist(withlabel=True, ndim=3)
    train_iter = iterators.SerialIterator(train[:1000],
                                          batch_size=32,
                                          shuffle=True)
    test_iter = iterators.SerialIterator(test[:1000],
                                         batch_size=32,
                                         shuffle=False,
                                         repeat=False)

    model = CapsNet()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter,
                                       optimizer,
                                       loss_func=model.loss)
    trainer = training.Trainer(updater,
                               stop_trigger=(10, "epoch"),
                               out=result_dir)
    trainer.extend(extensions.Evaluator(test_iter, model,
                                        eval_func=model.loss))
    trainer.extend(extensions.dump_graph("main/loss"))
    trainer.extend(
        extensions.snapshot(filename="snapshot_{.updater.epoch}.npz"),
        trigger=(1, "epoch")
    )
    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ["main/loss", "validation/main/loss"],
                "epoch", file_name="loss.png"
            )
        )
    trainer.extend(
        extensions.PrintReport(
            ["epoch", "main/loss", "main/c_loss", "main/r_loss",
             "validation/main/loss", "validation/main/c_loss",
             "validation/main/r_loss", "elapsed_time"]
        )
    )
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    model.save_structure(result_dir + "/model_structure.json")
    serializers.save_npz(result_dir + "/model_weights.npz", model)
