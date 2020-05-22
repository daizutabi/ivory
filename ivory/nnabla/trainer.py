from dataclasses import dataclass

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.logger import logger

import ivory.core.trainer

# context = get_extension_context("cudnn")
# context.backend[0]
# get_extension_context()


# @dataclass
# class Trainer(ivory.core.trainer.Trainer):
#     context_name: str = "cpu"
#     device_id: str = "0"
#     type_config: str = "float"
#     scheduler_step_mode: str = "epoch"
#
#     def on_fit_begin(self, run):
#         context = get_extension_context(
#             self.context, device_id=self.device_id, type_config=self.type_config
#         )
#         logger.info(f"Running in {context.backend[0]}")
#         nn.set_default_context(context)
#
#         run.dataloaders
#
#
#     def on_train_begin(self, run):
#         run.model.train()
#
#     def train_step(self, run, index, input, target):
#         if self.gpu:
#             input = utils.cuda(input)
#             target = utils.cuda(target)
#         output = self.forward(run.model, input)
#         if run.results:
#             run.results.step(index, output, target)
#         loss = run.metrics.step(input, output, target)
#         optimizer = run.optimizer
#         optimizer.zero_grad()
#         if self.gpu and self.precision == 16:
#             with amp.scale_loss(loss, optimizer) as scaled_loss:
#                 scaled_loss.backward()
#         else:
#             loss.backward()
#         optimizer.step()
#         if run.sheduler and self.scheduler_step_mode == "batch":
#             run.scheduler.step()
#
#     def on_val_begin(self, run):
#         run.model.eval()
#
#     @torch.no_grad()
#     def val_step(self, run, index, input, target):
#         if self.gpu:
#             input = utils.cuda(input)
#             target = utils.cuda(target)
#         output = self.forward(run.model, input)
#         if run.results:
#             run.results.step(index, output, target)
#         run.metrics.step(input, output, target)
#
#     def on_epoch_end(self, run):
#         if run.scheduler and self.scheduler_step_mode == "epoch":
#             if isinstance(run.scheduler, ReduceLROnPlateau):
#                 run.scheduler.step(run.monitor.score)
#             else:
#                 run.scheduler.step()
#
#     def on_test_begin(self, run):
#         self.on_fit_begin(run)
#         run.model.eval()
#
#     @torch.no_grad()
#     def test_step(self, run, index, input, *target):
#         if self.gpu:
#             input = utils.cuda(input)
#         output = self.forward(run.model, input)
#         if run.results:
#             run.results.step(index, output, *target)
#
#     def forward(self, model, input):
#         return model(input)
