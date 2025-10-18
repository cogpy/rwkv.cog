#include "rwkv.h"
#include "rwkv_opencog.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "ASSERTION FAILED at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            return 1; \
        } \
    } while(0)

#define ASSERT_FALSE(condition) ASSERT_TRUE(!(condition))
#define ASSERT_NOT_NULL(ptr) ASSERT_TRUE((ptr) != NULL)
#define ASSERT_NULL(ptr) ASSERT_TRUE((ptr) == NULL)
#define ASSERT_EQUAL(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NOT_EQUAL(a, b) ASSERT_TRUE((a) != (b))

int test_atomspace_basic_operations() {
    printf("Testing AtomSpace basic operations...\n");
    
    struct rwkv_atomspace * atomspace = rwkv_atomspace_create();
    ASSERT_NOT_NULL(atomspace);
    
    // Test initial state
    ASSERT_EQUAL(rwkv_atomspace_get_size(atomspace), 0);
    ASSERT_EQUAL(rwkv_atomspace_get_node_count(atomspace), 0);
    ASSERT_EQUAL(rwkv_atomspace_get_link_count(atomspace), 0);
    
    // Create some nodes
    rwkv_atom_handle_t cat_handle = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Cat"
    );
    ASSERT_NOT_EQUAL(cat_handle, RWKV_INVALID_ATOM_HANDLE);
    
    rwkv_atom_handle_t animal_handle = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Animal"
    );
    ASSERT_NOT_EQUAL(animal_handle, RWKV_INVALID_ATOM_HANDLE);
    
    // Test duplicate node creation
    rwkv_atom_handle_t cat_handle2 = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Cat"
    );
    ASSERT_EQUAL(cat_handle, cat_handle2);  // Should return the same handle
    
    // Check statistics
    ASSERT_EQUAL(rwkv_atomspace_get_size(atomspace), 2);
    ASSERT_EQUAL(rwkv_atomspace_get_node_count(atomspace), 2);
    ASSERT_EQUAL(rwkv_atomspace_get_link_count(atomspace), 0);
    
    rwkv_atomspace_free(atomspace);
    printf("AtomSpace basic operations: PASSED\n");
    return 0;
}

int test_atom_properties() {
    printf("Testing atom properties...\n");
    
    struct rwkv_atomspace * atomspace = rwkv_atomspace_create();
    ASSERT_NOT_NULL(atomspace);
    
    rwkv_atom_handle_t handle = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "TestConcept"
    );
    ASSERT_NOT_EQUAL(handle, RWKV_INVALID_ATOM_HANDLE);
    
    struct rwkv_atom * atom = rwkv_atomspace_get_atom(atomspace, handle);
    ASSERT_NOT_NULL(atom);
    
    // Test type
    ASSERT_EQUAL(rwkv_atom_get_type(atom), RWKV_ATOM_CONCEPT_NODE);
    
    // Test name
    const char * name = rwkv_atom_get_name(atom);
    ASSERT_NOT_NULL(name);
    ASSERT_TRUE(strcmp(name, "TestConcept") == 0);
    
    // Test truth value
    rwkv_truth_value_t tv_set = {0.8f, 0.9f};
    ASSERT_TRUE(rwkv_atom_set_truth_value(atom, &tv_set));
    
    rwkv_truth_value_t tv_get;
    ASSERT_TRUE(rwkv_atom_get_truth_value(atom, &tv_get));
    ASSERT_TRUE(fabs(tv_get.strength - 0.8f) < 0.001f);
    ASSERT_TRUE(fabs(tv_get.confidence - 0.9f) < 0.001f);
    
    // Test attention value
    rwkv_attention_value_t av_set = {1.5f, 0.3f, 0.1f};
    ASSERT_TRUE(rwkv_atom_set_attention_value(atom, &av_set));
    
    rwkv_attention_value_t av_get;
    ASSERT_TRUE(rwkv_atom_get_attention_value(atom, &av_get));
    ASSERT_TRUE(fabs(av_get.sti - 1.5f) < 0.001f);
    ASSERT_TRUE(fabs(av_get.lti - 0.3f) < 0.001f);
    ASSERT_TRUE(fabs(av_get.vlti - 0.1f) < 0.001f);
    
    rwkv_atomspace_free(atomspace);
    printf("Atom properties: PASSED\n");
    return 0;
}

int test_links() {
    printf("Testing links...\n");
    
    struct rwkv_atomspace * atomspace = rwkv_atomspace_create();
    ASSERT_NOT_NULL(atomspace);
    
    // Create nodes
    rwkv_atom_handle_t cat = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Cat"
    );
    rwkv_atom_handle_t animal = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Animal"
    );
    
    // Create inheritance link: Cat -> Animal
    rwkv_atom_handle_t outgoing[2] = {cat, animal};
    rwkv_atom_handle_t inheritance = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_INHERITANCE_LINK, outgoing, 2
    );
    ASSERT_NOT_EQUAL(inheritance, RWKV_INVALID_ATOM_HANDLE);
    
    // Test link properties
    struct rwkv_atom * link_atom = rwkv_atomspace_get_atom(atomspace, inheritance);
    ASSERT_NOT_NULL(link_atom);
    ASSERT_EQUAL(rwkv_atom_get_type(link_atom), RWKV_ATOM_INHERITANCE_LINK);
    
    // Test outgoing atoms
    rwkv_atom_handle_t retrieved_outgoing[10];
    size_t count = rwkv_atom_get_outgoing(link_atom, retrieved_outgoing, 10);
    ASSERT_EQUAL(count, 2);
    ASSERT_EQUAL(retrieved_outgoing[0], cat);
    ASSERT_EQUAL(retrieved_outgoing[1], animal);
    
    // Check statistics
    ASSERT_EQUAL(rwkv_atomspace_get_size(atomspace), 3);
    ASSERT_EQUAL(rwkv_atomspace_get_node_count(atomspace), 2);
    ASSERT_EQUAL(rwkv_atomspace_get_link_count(atomspace), 1);
    
    rwkv_atomspace_free(atomspace);
    printf("Links: PASSED\n");
    return 0;
}

int test_pattern_matching() {
    printf("Testing pattern matching...\n");
    
    struct rwkv_atomspace * atomspace = rwkv_atomspace_create();
    ASSERT_NOT_NULL(atomspace);
    
    // Create multiple concept nodes
    rwkv_atom_handle_t cat = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Cat"
    );
    rwkv_atom_handle_t dog = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Dog"
    );
    rwkv_atom_handle_t red = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_PREDICATE_NODE, "Red"
    );
    
    // Pattern match for concept nodes using cat as pattern
    rwkv_atom_handle_t results[10];
    size_t count = rwkv_atomspace_pattern_match(atomspace, cat, results, 10);
    
    ASSERT_EQUAL(count, 1);  // Should find only dog (not cat itself)
    ASSERT_EQUAL(results[0], dog);
    
    rwkv_atomspace_free(atomspace);
    printf("Pattern matching: PASSED\n");
    return 0;
}

int test_inference() {
    printf("Testing inference...\n");
    
    struct rwkv_atomspace * atomspace = rwkv_atomspace_create();
    ASSERT_NOT_NULL(atomspace);
    
    // Create: Cat -> Animal (implication)
    rwkv_atom_handle_t cat = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Cat"
    );
    rwkv_atom_handle_t animal = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Animal"
    );
    
    rwkv_atom_handle_t implication_outgoing[2] = {cat, animal};
    rwkv_atom_handle_t implication = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_IMPLICATION_LINK, implication_outgoing, 2
    );
    ASSERT_NOT_EQUAL(implication, RWKV_INVALID_ATOM_HANDLE);
    
    // Test forward inference
    rwkv_atom_handle_t conclusions[10];
    size_t num_conclusions;
    
    ASSERT_TRUE(rwkv_atomspace_forward_inference(
        atomspace, cat, conclusions, 10, &num_conclusions
    ));
    
    ASSERT_EQUAL(num_conclusions, 1);
    ASSERT_EQUAL(conclusions[0], animal);
    
    rwkv_atomspace_free(atomspace);
    printf("Inference: PASSED\n");
    return 0;
}

int test_rwkv_integration() {
    printf("Testing RWKV integration...\n");
    
    struct rwkv_atomspace * atomspace = rwkv_atomspace_create();
    ASSERT_NOT_NULL(atomspace);
    
    // Create a mock state array
    const size_t state_len = 100;
    float state[state_len];
    
    // Initialize with some test values
    for (size_t i = 0; i < state_len; i++) {
        state[i] = (i % 10 == 0) ? 0.5f * (i / 10) : 0.0f;
    }
    
    // Convert state to atoms
    ASSERT_TRUE(rwkv_context_to_atoms(
        NULL,  // rwkv_ctx not needed for this basic test
        atomspace,
        state,
        state_len
    ));
    
    // Check that atoms were created
    size_t atom_count_before = rwkv_atomspace_get_size(atomspace);
    ASSERT_TRUE(atom_count_before > 0);
    
    // Convert atoms back to state
    float recovered_state[state_len];
    ASSERT_TRUE(rwkv_atoms_to_context(
        atomspace,
        NULL,  // rwkv_ctx not needed for this basic test
        recovered_state,
        state_len
    ));
    
    // The recovered state won't be identical due to the simplified conversion,
    // but significant values should be preserved in some form
    int significant_preserved = 0;
    for (size_t i = 0; i < state_len; i++) {
        if (fabs(state[i]) > 0.1f && fabs(recovered_state[i]) > 0.0f) {
            significant_preserved++;
        }
    }
    ASSERT_TRUE(significant_preserved > 0);
    
    rwkv_atomspace_free(atomspace);
    printf("RWKV integration: PASSED\n");
    return 0;
}

int main() {
    printf("Running OpenCog Integration Tests...\n\n");
    
    int result = 0;
    
    result |= test_atomspace_basic_operations();
    result |= test_atom_properties();
    result |= test_links();
    result |= test_pattern_matching();
    result |= test_inference();
    result |= test_rwkv_integration();
    
    if (result == 0) {
        printf("\nAll OpenCog integration tests PASSED!\n");
    } else {
        printf("\nSome tests FAILED!\n");
    }
    
    return result;
}