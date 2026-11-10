"""Small real Chemprop fit with missing labels, checkpoint reload and epoch resume."""
import numpy as np
import pytest


def test_real_multitask_training_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, Callback
    from chemprop import data, models
    from expansionrx.multitask import make_model
    torch.set_num_threads(2)
    pl.seed_everything(42)
    smiles = ["CC", "CCC", "CCCC", "CCO", "CCCO", "CCN", "CCCN", "CCCl", "CCBr", "CCOC", "CCNC", "CCS"]
    ys = np.random.default_rng(42).normal(size=(12, 9))
    ys[::2, ::2] = np.nan
    ds = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s, y) for s, y in zip(smiles, ys)])
    loader = data.build_dataloader(ds, batch_size=4, num_workers=0, shuffle=False)
    folder = tmp_path / "checkpoints"

    class StopAfterFirstEpoch(Callback):
        def on_validation_end(self, trainer, pl_module):
            if trainer.current_epoch == 0:
                trainer.should_stop = True

    def make_trainer(stop=False):
        checkpoint = ModelCheckpoint(dirpath=folder, filename="best", monitor="val_loss", mode="min",
                                     save_top_k=1, save_last=True, enable_version_counter=False)
        callbacks = [checkpoint] + ([StopAfterFirstEpoch()] if stop else [])
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, callbacks=callbacks,
                             logger=False, enable_progress_bar=False, num_sanity_val_steps=0)
        return trainer

    first = make_trainer(stop=True)
    first.fit(make_model(hidden_size=16, epochs=2), loader, loader)
    initial = torch.load(folder / "last.ckpt", map_location="cpu", weights_only=False)
    assert initial["epoch"] == 0
    assert "optimizer_states" in initial and "lr_schedulers" in initial
    resumed = make_trainer()
    resumed.fit(make_model(hidden_size=16, epochs=2), loader, loader, ckpt_path=str(folder / "last.ckpt"))
    final = torch.load(folder / "last.ckpt", map_location="cpu", weights_only=False)
    assert final["epoch"] == 1
    assert final["global_step"] > initial["global_step"]
    restored = models.MPNN.load_from_checkpoint(folder / "best.ckpt", map_location="cpu", weights_only=False)
    pred = torch.cat(resumed.predict(restored, dataloaders=loader))
    assert pred.shape == (12, 9)
    assert torch.isfinite(pred).all()
