from autofaiss import build_index

from config import BASE_PATH, ID_MAPPING, INDEX_NAME, OUTPUT_DIR

build_index(
    embeddings=f"{BASE_PATH}/{OUTPUT_DIR}",
    index_path=f"{BASE_PATH}/{INDEX_NAME}",
    index_infos_path=f"{BASE_PATH}/index_infos.json",
    file_format="parquet",
    embedding_column_name="embedding",
    id_columns=["url"],
    ids_path=f"{BASE_PATH}/{ID_MAPPING}",
)
