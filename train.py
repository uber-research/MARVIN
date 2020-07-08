'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
