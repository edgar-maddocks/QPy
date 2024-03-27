from finrl import StocksEnv, DQNAgent, Model
import yfinance as yf

if __name__ == "__main__":
    ticker = yf.Ticker("GOOGL")
    data = ticker.history(period="1Y")

    env = StocksEnv(data=data)
    net = Model(
        env.observation_space.shape, env.action_space.n, neurons=512, device="cuda"
    )
    agent = DQNAgent(
        env, net, gamma=0.75, epsilon_decay=0.9999, max_mem_length=8192, device="cuda"
    )

    agent.train(
        5000,
        1,
        min_mem_size=2048,
        batch_size=256,
        update_target_interval=50000,
        use_wandb=True,
    )