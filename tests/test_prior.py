import torch

from resemreason.types import KGPath, RelationSchema


def test_path_prior_normalizes(pipeline):
    paths = [
        KGPath(
            nodes=("disease:ra", "drug:methotrexate", "effect:nausea"),
            relations=("indication", "side_effect"),
            directions=(-1, 1),
        ),
        KGPath(
            nodes=("disease:ra", "process:inflammation"),
            relations=("associated_with",),
            directions=(1,),
        ),
    ]
    schema = RelationSchema(
        hops=2,
        entity_types=("disease", "drug", "effect"),
        relations=("indication", "side_effect"),
        directions=(-1, 1),
        probability=1.0,
    )
    _, _, topology = pipeline.model.path_encoder.encode_paths(paths, pipeline.device)
    scores, prior = pipeline.model.path_prior(
        paths, [schema], topology, pipeline.model.path_encoder
    )
    assert scores.shape == (2,)
    assert torch.allclose(prior.sum(), torch.tensor(1.0), atol=1e-6)
    assert prior[0] > prior[1]
