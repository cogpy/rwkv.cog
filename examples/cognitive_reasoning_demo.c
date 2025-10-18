/**
 * OpenCog Integration Demo for RWKV.cpp
 * 
 * This example demonstrates how to use the OpenCog cognitive architecture
 * features integrated into RWKV.cpp for knowledge representation and reasoning.
 */

#include "rwkv.h"
#include "rwkv_opencog.h"
#include <stdio.h>
#include <stdlib.h>

static void print_separator(const char* title) {
    printf("\n=== %s ===\n", title);
}

static void print_atom_info(struct rwkv_atom* atom) {
    if (!atom) return;
    
    rwkv_atom_type_t type = rwkv_atom_get_type(atom);
    const char* name = rwkv_atom_get_name(atom);
    
    rwkv_truth_value_t tv;
    rwkv_attention_value_t av;
    rwkv_atom_get_truth_value(atom, &tv);
    rwkv_atom_get_attention_value(atom, &av);
    
    printf("  Type: %d, Name: %s, TV: (%.2f, %.2f), AV: (%.2f, %.2f, %.2f)\n",
           type, name ? name : "NULL", tv.strength, tv.confidence,
           av.sti, av.lti, av.vlti);
}

int main() {
    printf("RWKV OpenCog Cognitive Architecture Demo\n");
    printf("========================================\n");
    
    // 1. Create AtomSpace for knowledge representation
    print_separator("Creating AtomSpace");
    struct rwkv_atomspace* atomspace = rwkv_atomspace_create();
    if (!atomspace) {
        fprintf(stderr, "Failed to create AtomSpace\n");
        return 1;
    }
    printf("AtomSpace created successfully\n");
    
    // 2. Build a simple knowledge base about animals
    print_separator("Building Knowledge Base");
    
    // Create concept nodes
    rwkv_atom_handle_t cat = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Cat");
    rwkv_atom_handle_t dog = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Dog");
    rwkv_atom_handle_t animal = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Animal");
    rwkv_atom_handle_t mammal = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_CONCEPT_NODE, "Mammal");
    
    // Create predicate nodes
    rwkv_atom_handle_t has_fur = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_PREDICATE_NODE, "HasFur");
    rwkv_atom_handle_t warm_blooded = rwkv_atomspace_add_node(
        atomspace, RWKV_ATOM_PREDICATE_NODE, "WarmBlooded");
    
    printf("Created %zu concept and predicate nodes\n", rwkv_atomspace_get_node_count(atomspace));
    
    // 3. Create inheritance relationships
    print_separator("Creating Inheritance Links");
    
    rwkv_atom_handle_t cat_animal_outgoing[] = {cat, animal};
    rwkv_atom_handle_t cat_animal_link = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_INHERITANCE_LINK, cat_animal_outgoing, 2);
    
    rwkv_atom_handle_t dog_animal_outgoing[] = {dog, animal};
    rwkv_atom_handle_t dog_animal_link = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_INHERITANCE_LINK, dog_animal_outgoing, 2);
    
    rwkv_atom_handle_t animal_mammal_outgoing[] = {animal, mammal};
    rwkv_atom_handle_t animal_mammal_link = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_INHERITANCE_LINK, animal_mammal_outgoing, 2);
    
    printf("Created inheritance links: Cat->Animal, Dog->Animal, Animal->Mammal\n");
    
    // 4. Create evaluation links for properties
    print_separator("Creating Property Evaluations");
    
    // Mammals have fur
    rwkv_atom_handle_t mammal_fur_list_outgoing[] = {mammal, has_fur};
    rwkv_atom_handle_t mammal_fur_list = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_LIST_LINK, mammal_fur_list_outgoing, 2);
    
    rwkv_atom_handle_t mammal_fur_eval_outgoing[] = {has_fur, mammal_fur_list};
    rwkv_atom_handle_t mammal_fur_eval = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_EVALUATION_LINK, mammal_fur_eval_outgoing, 2);
    
    // Set truth values for facts
    struct rwkv_atom* mammal_fur_atom = rwkv_atomspace_get_atom(atomspace, mammal_fur_eval);
    if (mammal_fur_atom) {
        rwkv_truth_value_t strong_belief = {0.9f, 0.8f};  // High strength, high confidence
        rwkv_atom_set_truth_value(mammal_fur_atom, &strong_belief);
    }
    
    printf("Created evaluation: Mammals have fur (strength=0.9, confidence=0.8)\n");
    
    // 5. Create implication for reasoning
    print_separator("Creating Implication Rules");
    
    // If X inherits from Animal, and Animal inherits from Mammal, then X inherits from Mammal
    rwkv_atom_handle_t transitivity_premise_outgoing[] = {cat, animal};
    rwkv_atom_handle_t transitivity_premise = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_INHERITANCE_LINK, transitivity_premise_outgoing, 2);
    
    rwkv_atom_handle_t transitivity_conclusion_outgoing[] = {cat, mammal};
    rwkv_atom_handle_t transitivity_conclusion = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_INHERITANCE_LINK, transitivity_conclusion_outgoing, 2);
    
    rwkv_atom_handle_t implication_outgoing[] = {transitivity_premise, transitivity_conclusion};
    rwkv_atom_handle_t implication = rwkv_atomspace_add_link(
        atomspace, RWKV_ATOM_IMPLICATION_LINK, implication_outgoing, 2);
    
    printf("Created implication rule for transitive inheritance\n");
    
    // 6. Display AtomSpace statistics
    print_separator("AtomSpace Statistics");
    printf("Total atoms: %zu\n", rwkv_atomspace_get_size(atomspace));
    printf("Nodes: %zu\n", rwkv_atomspace_get_node_count(atomspace));
    printf("Links: %zu\n", rwkv_atomspace_get_link_count(atomspace));
    
    // 7. Demonstrate pattern matching
    print_separator("Pattern Matching Demo");
    
    rwkv_atom_handle_t concept_matches[10];
    size_t match_count = rwkv_atomspace_pattern_match(
        atomspace, cat, concept_matches, 10);
    
    printf("Found %zu concept nodes matching Cat's type:\n", match_count);
    for (size_t i = 0; i < match_count; i++) {
        struct rwkv_atom* matched_atom = rwkv_atomspace_get_atom(atomspace, concept_matches[i]);
        if (matched_atom) {
            printf("  - %s\n", rwkv_atom_get_name(matched_atom));
        }
    }
    
    // 8. Demonstrate inference
    print_separator("Forward Inference Demo");
    
    rwkv_atom_handle_t conclusions[10];
    size_t num_conclusions;
    
    bool inference_result = rwkv_atomspace_forward_inference(
        atomspace, transitivity_premise, conclusions, 10, &num_conclusions);
    
    if (inference_result && num_conclusions > 0) {
        printf("Forward inference from Cat->Animal produced %zu conclusions:\n", num_conclusions);
        for (size_t i = 0; i < num_conclusions; i++) {
            struct rwkv_atom* conclusion_atom = rwkv_atomspace_get_atom(atomspace, conclusions[i]);
            if (conclusion_atom) {
                // This would be a link, so we need to examine its outgoing atoms
                rwkv_atom_handle_t outgoing_atoms[10];
                size_t outgoing_count = rwkv_atom_get_outgoing(conclusion_atom, outgoing_atoms, 10);
                
                if (outgoing_count >= 2) {
                    struct rwkv_atom* from_atom = rwkv_atomspace_get_atom(atomspace, outgoing_atoms[0]);
                    struct rwkv_atom* to_atom = rwkv_atomspace_get_atom(atomspace, outgoing_atoms[1]);
                    
                    if (from_atom && to_atom) {
                        printf("  - %s inherits from %s\n", 
                               rwkv_atom_get_name(from_atom), 
                               rwkv_atom_get_name(to_atom));
                    }
                }
            }
        }
    } else {
        printf("No conclusions drawn from forward inference\n");
    }
    
    // 9. Simulate RWKV integration
    print_separator("RWKV Integration Simulation");
    
    // Create a mock RWKV state with some activations
    const size_t state_len = 50;
    float mock_state[50];
    
    // Initialize with some pattern (simulate language model activations)
    for (size_t i = 0; i < state_len; i++) {
        mock_state[i] = (i % 5 == 0) ? 0.3f * (i / 5) : 0.05f * i;
    }
    
    printf("Converting RWKV state to cognitive atoms...\n");
    bool conversion_success = rwkv_context_to_atoms(
        NULL,  // No actual RWKV context for this demo
        atomspace,
        mock_state,
        state_len
    );
    
    if (conversion_success) {
        printf("Successfully converted %zu state elements to cognitive representation\n", state_len);
        printf("AtomSpace now contains %zu atoms\n", rwkv_atomspace_get_size(atomspace));
        
        // Convert back to state
        float recovered_state[50];
        bool recovery_success = rwkv_atoms_to_context(
            atomspace,
            NULL,  // No actual RWKV context for this demo
            recovered_state,
            state_len
        );
        
        if (recovery_success) {
            printf("Successfully recovered state from cognitive atoms\n");
            
            // Compare some values
            printf("Sample state comparison (first 10 elements):\n");
            printf("  Original -> Recovered\n");
            for (size_t i = 0; i < 10 && i < state_len; i++) {
                printf("  %.3f -> %.3f\n", mock_state[i], recovered_state[i]);
            }
        }
    }
    
    // 10. Memory consolidation demo
    print_separator("Memory Consolidation");
    
    size_t atoms_before_consolidation = rwkv_atomspace_get_size(atomspace);
    bool consolidation_result = rwkv_atomspace_consolidate_memory(atomspace, 0.8f);
    size_t atoms_after_consolidation = rwkv_atomspace_get_size(atomspace);
    
    printf("Memory consolidation %s\n", consolidation_result ? "succeeded" : "failed");
    printf("Atoms before: %zu, after: %zu\n", atoms_before_consolidation, atoms_after_consolidation);
    
    // 11. Final statistics and cleanup
    print_separator("Final Statistics & Cleanup");
    
    printf("Final AtomSpace contents:\n");
    printf("  Total atoms: %zu\n", rwkv_atomspace_get_size(atomspace));
    printf("  Concept nodes: %zu\n", rwkv_atomspace_get_node_count(atomspace));
    printf("  Links: %zu\n", rwkv_atomspace_get_link_count(atomspace));
    
    // Clean up
    rwkv_atomspace_free(atomspace);
    printf("\nAtomSpace freed. Demo completed successfully!\n");
    
    printf("\n=== SUMMARY ===\n");
    printf("This demo showed:\n");
    printf("1. AtomSpace creation and management\n");
    printf("2. Knowledge representation with nodes and links\n");
    printf("3. Truth and attention values for uncertain reasoning\n");
    printf("4. Pattern matching for knowledge retrieval\n");
    printf("5. Forward inference for logical deduction\n");
    printf("6. Integration between RWKV language model and cognitive architecture\n");
    printf("7. Memory consolidation for efficient knowledge management\n");
    
    return 0;
}