#ifndef RWKV_OPENCOG_H
#define RWKV_OPENCOG_H

#include "rwkv.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// OpenCog-style AtomSpace integration for RWKV

// Forward declarations
struct rwkv_atom;
struct rwkv_atomspace;

// Atom types based on OpenCog taxonomy
typedef enum {
    RWKV_ATOM_NODE,
    RWKV_ATOM_LINK,
    RWKV_ATOM_CONCEPT_NODE,
    RWKV_ATOM_PREDICATE_NODE,
    RWKV_ATOM_NUMBER_NODE,
    RWKV_ATOM_VARIABLE_NODE,
    RWKV_ATOM_LIST_LINK,
    RWKV_ATOM_EVALUATION_LINK,
    RWKV_ATOM_IMPLICATION_LINK,
    RWKV_ATOM_AND_LINK,
    RWKV_ATOM_OR_LINK,
    RWKV_ATOM_NOT_LINK,
    RWKV_ATOM_SIMILARITY_LINK,
    RWKV_ATOM_INHERITANCE_LINK
} rwkv_atom_type_t;

// Truth value representation (simplified OpenCog truth values)
typedef struct {
    float strength;    // [0.0, 1.0] - strength of belief
    float confidence;  // [0.0, 1.0] - confidence in the belief
} rwkv_truth_value_t;

// Attention value for cognitive focus (simplified)
typedef struct {
    float sti;  // Short-term importance
    float lti;  // Long-term importance
    float vlti; // Very long-term importance
} rwkv_attention_value_t;

// Atom handle (unique identifier)
typedef uint64_t rwkv_atom_handle_t;
#define RWKV_INVALID_ATOM_HANDLE 0

// Create a new AtomSpace for cognitive operations
RWKV_API struct rwkv_atomspace * rwkv_atomspace_create(void);

// Free an AtomSpace
RWKV_API void rwkv_atomspace_free(struct rwkv_atomspace * atomspace);

// Create a node atom with given type and name
RWKV_API rwkv_atom_handle_t rwkv_atomspace_add_node(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_type_t type,
    const char * name
);

// Create a link atom connecting multiple atoms
RWKV_API rwkv_atom_handle_t rwkv_atomspace_add_link(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_type_t type,
    const rwkv_atom_handle_t * outgoing,
    size_t outgoing_count
);

// Get atom by handle
RWKV_API struct rwkv_atom * rwkv_atomspace_get_atom(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_handle_t handle
);

// Set truth value for an atom
RWKV_API bool rwkv_atom_set_truth_value(
    struct rwkv_atom * atom,
    const rwkv_truth_value_t * tv
);

// Get truth value of an atom
RWKV_API bool rwkv_atom_get_truth_value(
    struct rwkv_atom * atom,
    rwkv_truth_value_t * tv
);

// Set attention value for an atom
RWKV_API bool rwkv_atom_set_attention_value(
    struct rwkv_atom * atom,
    const rwkv_attention_value_t * av
);

// Get attention value of an atom
RWKV_API bool rwkv_atom_get_attention_value(
    struct rwkv_atom * atom,
    rwkv_attention_value_t * av
);

// Get atom type
RWKV_API rwkv_atom_type_t rwkv_atom_get_type(struct rwkv_atom * atom);

// Get atom name (for nodes)
RWKV_API const char * rwkv_atom_get_name(struct rwkv_atom * atom);

// Get outgoing atoms (for links)
RWKV_API size_t rwkv_atom_get_outgoing(
    struct rwkv_atom * atom,
    rwkv_atom_handle_t * outgoing,
    size_t max_count
);

// Pattern matching: find atoms matching a pattern
RWKV_API size_t rwkv_atomspace_pattern_match(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_handle_t pattern,
    rwkv_atom_handle_t * results,
    size_t max_results
);

// Integration with RWKV language model

// Convert RWKV context state to cognitive atoms
RWKV_API bool rwkv_context_to_atoms(
    struct rwkv_context * rwkv_ctx,
    struct rwkv_atomspace * atomspace,
    const float * state,
    size_t state_len
);

// Query AtomSpace for language generation context
RWKV_API bool rwkv_atoms_to_context(
    struct rwkv_atomspace * atomspace,
    struct rwkv_context * rwkv_ctx,
    float * state,
    size_t state_len
);

// Cognitive reasoning: simple inference engine
RWKV_API bool rwkv_atomspace_forward_inference(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_handle_t premise,
    rwkv_atom_handle_t * conclusions,
    size_t max_conclusions,
    size_t * num_conclusions
);

// Memory consolidation: merge similar concepts
RWKV_API bool rwkv_atomspace_consolidate_memory(
    struct rwkv_atomspace * atomspace,
    float similarity_threshold
);

// Statistics and introspection
RWKV_API size_t rwkv_atomspace_get_size(struct rwkv_atomspace * atomspace);
RWKV_API size_t rwkv_atomspace_get_node_count(struct rwkv_atomspace * atomspace);
RWKV_API size_t rwkv_atomspace_get_link_count(struct rwkv_atomspace * atomspace);

#ifdef __cplusplus
}
#endif

#endif // RWKV_OPENCOG_H