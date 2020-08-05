'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
This is the main training file for the system. This will create the trainer object
and call it.
'''

from marvin.trainers.IL import ILTrainer
from marvin.trainers.RL import RLTrainer
from marvin.utils.trainer_parameters import parser

if __name__ == "__main__":
    # args are the input hyperparameters and details that the user sets
    args = parser.parse_args()
    if args.rl:
        t = RLTrainer(args)
    else:
        t = ILTrainer(args)

    t.train()
