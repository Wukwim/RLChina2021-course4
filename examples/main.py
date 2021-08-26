from runner import Runner
import argparse
import os

if __name__ == '__main__':
    # set env and algo
    if os.path.exists('Gt.txt'):
        os.remove('Gt.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default="snakes_2p", type=str)
    parser.add_argument('--algo', default="iql", type=str,
                        help="tabularq/sarsa/dqn/ppo/ddpg/ac/ddqn/duelingq/sac/pg/sac/td3")

    parser.add_argument('--reload_config', action='store_true')  # 加是true；不加为false
    args = parser.parse_args()
    # args.reload_config = False

    print("================== args: ", args)
    print("== args.reload_config: ", args.reload_config)

    runner = Runner(args)
    runner.run()