# RWKV.cpp OpenCog Integration

This document describes the OpenCog cognitive architecture integration added to RWKV.cpp, providing advanced knowledge representation, reasoning, and cognitive capabilities to complement RWKV's language modeling.

## Overview

The OpenCog integration brings a powerful hypergraph-based knowledge representation system (AtomSpace) to RWKV.cpp, enabling:

- **Knowledge Representation**: Store and manipulate structured knowledge using atoms (nodes and links)
- **Uncertain Reasoning**: Truth values and attention mechanisms for handling uncertainty
- **Pattern Matching**: Find atoms that match specific patterns or queries  
- **Logical Inference**: Forward chaining inference engine for deductive reasoning
- **Cognitive Integration**: Bridge between RWKV language model states and symbolic knowledge

## Core Concepts

### AtomSpace

The AtomSpace is a hypergraph database for storing and retrieving knowledge. It contains:

- **Nodes**: Represent concepts, predicates, variables, or values
- **Links**: Represent relationships between other atoms
- **Truth Values**: Strength and confidence for uncertain reasoning
- **Attention Values**: Short-term, long-term, and very long-term importance

### Atom Types

#### Node Types
- `RWKV_ATOM_CONCEPT_NODE`: Represents concepts (e.g., "Cat", "Animal")
- `RWKV_ATOM_PREDICATE_NODE`: Represents predicates (e.g., "HasFur", "Flies")  
- `RWKV_ATOM_NUMBER_NODE`: Represents numeric values
- `RWKV_ATOM_VARIABLE_NODE`: Represents variables in patterns

#### Link Types
- `RWKV_ATOM_INHERITANCE_LINK`: Represents inheritance (e.g., Cat → Animal)
- `RWKV_ATOM_EVALUATION_LINK`: Represents predicate evaluations
- `RWKV_ATOM_IMPLICATION_LINK`: Represents logical implications (if-then)
- `RWKV_ATOM_LIST_LINK`: Represents ordered collections
- `RWKV_ATOM_AND_LINK`, `RWKV_ATOM_OR_LINK`, `RWKV_ATOM_NOT_LINK`: Logical operators

## API Reference

### AtomSpace Management

```c
// Create a new AtomSpace
struct rwkv_atomspace * rwkv_atomspace_create(void);

// Free an AtomSpace
void rwkv_atomspace_free(struct rwkv_atomspace * atomspace);

// Get statistics
size_t rwkv_atomspace_get_size(struct rwkv_atomspace * atomspace);
size_t rwkv_atomspace_get_node_count(struct rwkv_atomspace * atomspace);  
size_t rwkv_atomspace_get_link_count(struct rwkv_atomspace * atomspace);
```

### Atom Creation

```c
// Create a node atom
rwkv_atom_handle_t rwkv_atomspace_add_node(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_type_t type,
    const char * name
);

// Create a link atom
rwkv_atom_handle_t rwkv_atomspace_add_link(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_type_t type,
    const rwkv_atom_handle_t * outgoing,
    size_t outgoing_count
);
```

### Atom Properties

```c
// Truth values (strength ∈ [0,1], confidence ∈ [0,1])
typedef struct {
    float strength;    // Strength of belief
    float confidence;  // Confidence in the belief  
} rwkv_truth_value_t;

bool rwkv_atom_set_truth_value(struct rwkv_atom * atom, const rwkv_truth_value_t * tv);
bool rwkv_atom_get_truth_value(struct rwkv_atom * atom, rwkv_truth_value_t * tv);

// Attention values (importance measures)
typedef struct {
    float sti;  // Short-term importance
    float lti;  // Long-term importance  
    float vlti; // Very long-term importance
} rwkv_attention_value_t;

bool rwkv_atom_set_attention_value(struct rwkv_atom * atom, const rwkv_attention_value_t * av);
bool rwkv_atom_get_attention_value(struct rwkv_atom * atom, rwkv_attention_value_t * av);
```

### Reasoning & Querying

```c
// Pattern matching
size_t rwkv_atomspace_pattern_match(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_handle_t pattern,
    rwkv_atom_handle_t * results,
    size_t max_results
);

// Forward inference
bool rwkv_atomspace_forward_inference(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_handle_t premise,
    rwkv_atom_handle_t * conclusions,
    size_t max_conclusions,
    size_t * num_conclusions
);
```

### RWKV Integration

```c
// Convert RWKV state to cognitive atoms
bool rwkv_context_to_atoms(
    struct rwkv_context * rwkv_ctx,
    struct rwkv_atomspace * atomspace,
    const float * state,
    size_t state_len
);

// Query AtomSpace for language generation context
bool rwkv_atoms_to_context(
    struct rwkv_atomspace * atomspace,
    struct rwkv_context * rwkv_ctx,
    float * state,
    size_t state_len
);
```

## Usage Examples

### Basic Knowledge Representation

```c
#include "rwkv.h"

struct rwkv_atomspace * atomspace = rwkv_atomspace_create();

// Create concept nodes
rwkv_atom_handle_t cat = rwkv_atomspace_add_node(
    atomspace, RWKV_ATOM_CONCEPT_NODE, "Cat");
rwkv_atom_handle_t animal = rwkv_atomspace_add_node(
    atomspace, RWKV_ATOM_CONCEPT_NODE, "Animal");

// Create inheritance link: Cat → Animal  
rwkv_atom_handle_t inheritance_outgoing[] = {cat, animal};
rwkv_atom_handle_t inheritance = rwkv_atomspace_add_link(
    atomspace, RWKV_ATOM_INHERITANCE_LINK, inheritance_outgoing, 2);

rwkv_atomspace_free(atomspace);
```

### Truth Values and Uncertainty

```c
struct rwkv_atom * atom = rwkv_atomspace_get_atom(atomspace, handle);

// Set truth value: 80% strength, 90% confidence
rwkv_truth_value_t tv = {0.8f, 0.9f};
rwkv_atom_set_truth_value(atom, &tv);

// Set attention value
rwkv_attention_value_t av = {1.5f, 0.3f, 0.1f};  // High STI
rwkv_atom_set_attention_value(atom, &av);
```

### Logical Reasoning

```c
// Create implication: If X is a Cat, then X is an Animal
rwkv_atom_handle_t implication_outgoing[] = {cat_premise, animal_conclusion};
rwkv_atom_handle_t implication = rwkv_atomspace_add_link(
    atomspace, RWKV_ATOM_IMPLICATION_LINK, implication_outgoing, 2);

// Forward inference from premise
rwkv_atom_handle_t conclusions[10];
size_t num_conclusions;
rwkv_atomspace_forward_inference(
    atomspace, cat_premise, conclusions, 10, &num_conclusions);
```

### RWKV-Cognitive Integration

```c
struct rwkv_context * rwkv_ctx = rwkv_init_from_file("model.bin", 4, 0);
struct rwkv_atomspace * atomspace = rwkv_atomspace_create();

// Evaluate RWKV and convert state to cognitive representation
float state[4096];
rwkv_eval(rwkv_ctx, token, NULL, state, NULL);
rwkv_context_to_atoms(rwkv_ctx, atomspace, state, 4096);

// Perform reasoning on cognitive representation
// ... pattern matching, inference, etc ...

// Convert back to RWKV state for continued generation
rwkv_atoms_to_context(atomspace, rwkv_ctx, state, 4096);
```

## Building and Testing

The OpenCog integration is automatically built with RWKV.cpp:

```bash
# Build with OpenCog integration
cmake .
cmake --build . --config Release

# Run tests (including OpenCog integration test)
ctest --output-on-failure

# Run cognitive reasoning demo
./bin/cognitive_reasoning_demo
```

## File Structure

- `rwkv_opencog.h` - OpenCog integration API header
- `rwkv_opencog.cpp` - OpenCog integration implementation  
- `tests/test_opencog_integration.c` - Comprehensive test suite
- `examples/cognitive_reasoning_demo.c` - Interactive demonstration

## Performance Considerations

### Memory Usage
- AtomSpace uses C++ containers for efficient storage
- Atoms are reference-counted and deduplicated automatically
- Memory usage scales with knowledge base size

### Concurrency
- AtomSpace operations are thread-safe using mutexes
- Multiple RWKV contexts can share a single AtomSpace
- Pattern matching and inference can run concurrently

### Optimization Tips
- Use attention values to focus on important atoms
- Implement memory consolidation for large knowledge bases
- Cache frequently accessed patterns
- Batch atom creation for better performance

## Integration Patterns

### Conversational AI
```c
// Before each response generation:
// 1. Convert current RWKV state to atoms
// 2. Perform reasoning on conversation context  
// 3. Update knowledge with new information
// 4. Convert enriched context back to RWKV state
```

### Knowledge-Augmented Generation
```c
// Query-time knowledge integration:
// 1. Pattern match user query against knowledge base
// 2. Retrieve relevant facts and inferences
// 3. Incorporate into language model context
// 4. Generate informed responses
```

### Continual Learning
```c
// Post-generation learning:
// 1. Extract facts from generated text
// 2. Add to knowledge base with appropriate truth values
// 3. Perform consistency checking and consolidation
// 4. Update attention values based on usage
```

## Advanced Features

### Custom Atom Types
Extend the system by adding new atom types for domain-specific knowledge representation.

### Pluggable Inference Engines  
The forward inference system can be extended with additional reasoning algorithms.

### Distributed AtomSpaces
For large-scale applications, AtomSpaces can be distributed across multiple nodes.

## Limitations and Future Work

### Current Limitations
- Simplified inference engine (forward chaining only)
- Basic pattern matching (exact type matching)
- Limited probabilistic reasoning capabilities
- No persistent storage (memory-only)

### Planned Enhancements
- Probabilistic Logic Networks (PLN) integration
- Advanced pattern matching with variables
- Persistent storage backends
- Distributed reasoning capabilities
- More sophisticated attention allocation

## References

- [OpenCog Framework](https://opencog.org/)
- [AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
- [RWKV Architecture Paper](https://arxiv.org/abs/2305.13048)
- [Cognitive Architecture Principles](https://wiki.opencog.org/w/CogPrime_Overview)

This integration represents a significant step toward building truly cognitive AI systems that can reason about their knowledge while maintaining the efficiency of the RWKV architecture.