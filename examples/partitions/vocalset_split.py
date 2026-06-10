from scripts.config.partition_config import PartitionConfig

partition = PartitionConfig(
    name="vocalset_singer_independent",
    train_singer_ids=[f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 7)],
    test_singer_ids=["f10", "f11", "m7", "m8", "m9"],
    singer_column="singer",
)
