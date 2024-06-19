collection.delete()

# Get the number of entities (vectors) in the collection
num_entities = collection.num_entities
# Print the number of vectors
print("Number of vectors in the collection:", num_entities)

vector_ids_to_delete = [1, 2, 3]  # Replace with your actual vector IDs
# Delete entities from the collection by their IDs
collection.delete_entity_by_id(vector_ids_to_delete)
# Confirm deletion
print("Entities with IDs {} have been deleted from the collection.".format(vector_ids_to_delete))