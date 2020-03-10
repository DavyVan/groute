// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#ifdef HAVE_METIS
#include <metis.h>
#endif
#include <nvgraph.h>

#include <unordered_set>
#include <groute/graphs/csr_graph.h>
#include <cmath>

namespace groute {
namespace graphs {

    namespace multi
    {
        MetisPartitioner::MetisPartitioner(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting METIS partitioning\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            // Convert to 64-bit for metis, idx_t is defined in metis.h
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, calling METIS\n", (int)IDXTYPEWIDTH);
            
            int result = METIS_PartGraphKway(
                &nnodes,                      // 
                &ncons,                       //
                row_start.data(),     //
                edge_dst.data(),      //
                NULL,                         // vwgt
                NULL,                         // vsize
                m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  // adjwgt
                &nparts,                      // nparts
                NULL,                         // tpwgts
                NULL,                         // ubvec
                NULL,                         // options
                &edgecut,                     // objval
                &partition_table[0]);         // part [out]

            if (result != METIS_OK) {
                printf(
                    "METIS partitioning failed (%s error), Exiting.\n", 
                    result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
                exit(0);
            }

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("METIS partitioning done\n");
#endif
        }

        void MetisPartitioner::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> MetisPartitioner::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

        std::vector<index_t> GetUniqueHalos(
            const index_t* edge_dst,
            index_t seg_snode, index_t seg_nnodes,
            index_t seg_sedge, index_t seg_nedges, int& halos_counter)
        {
            std::unordered_set<index_t> halos_set;
            halos_counter = 0;

            for (int i = 0; i < seg_nedges; ++i)
            {
                index_t dest = edge_dst[seg_sedge + i];
                if (dest < seg_snode || dest >= (seg_snode + seg_nnodes)) // an halo
                {
                    ++halos_counter;
                    halos_set.insert(dest);
                }
            }

            std::vector<index_t> halos_vec(halos_set.size());
            std::copy(halos_set.begin(), halos_set.end(), halos_vec.begin());

            return halos_vec;
        }

        // FQ
        NaivePartitioner::NaivePartitioner(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting naive partitioning\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, doing naive partitioning\n", (int)IDXTYPEWIDTH);
            
            // int result = METIS_PartGraphKway(
            //     &nnodes,                      // 
            //     &ncons,                       //
            //     row_start.data(),     //
            //     edge_dst.data(),      //
            //     NULL,                         //
            //     NULL,                         //
            //     m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  //
            //     &nparts,                      //
            //     NULL,                         //
            //     NULL,                         //
            //     NULL,                         //
            //     &edgecut,                     //
            //     &partition_table[0]);         //

            // FQ: we only need to modify partition_table
            idx_t nnodes_per_seg = nnodes / nparts;
            idx_t _t = 0;
            for (idx_t i = 0; i < nnodes; i++)
            {
                _t = i / nnodes_per_seg;
                if (_t >= nparts-1)
                    partition_table[i] = nparts-1;
                else
                    partition_table[i] = i / nnodes_per_seg;
                // printf("%d", partition_table[i]);
            }

            for (idx_t i = 0; i < 10; i++)
                printf("%d", partition_table[i]);

            printf("Building partitioned graph and lookup tables\n");

            // FQ: This struct store the relation between node ID and it partition ID
            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);    // FQ: allocate 1 such struct for every node

            // FQ: init the data
            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            // FQ: sort, put nodes belong to one partition together
            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            // FQ: reorganize partitioned graph in CSR
            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;    // FQ: record the boundary node, the first node in seg
                }

                // FQ: construct a lookup table between old and new node ID, because we sort the node_partitions
                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;       

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("Naive partitioning done\n");
#endif
        }

        void NaivePartitioner::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> NaivePartitioner::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

        MetisPartitionerDegreeW::MetisPartitionerDegreeW(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
#ifndef HAVE_METIS
            printf("\nWARNING: Binary not built with METIS support. Exiting.\n");
            exit(100);
#else
            printf("\nStarting METIS partitioning with vertex weights (degree)\n");

            idx_t nnodes = m_origin_graph.nnodes;
            idx_t nedges = m_origin_graph.nedges;

            idx_t ncons = 1;
            idx_t nparts = m_nsegs;

            printf("0..");
            fflush(stdout);
            idx_t edgecut;
            std::vector<idx_t> partition_table(nnodes);

            printf("1..");
            fflush(stdout);

            // Convert to 64-bit for metis
            std::vector<idx_t> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<idx_t>(m_origin_graph.row_start[i]);
            printf("2..");fflush(stdout);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<idx_t>(m_origin_graph.edge_dst[i]);
            printf("3..");fflush(stdout);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<idx_t>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, computing degrees\n", (int)IDXTYPEWIDTH);

            idx_t *vdegrees = (idx_t*) malloc(sizeof(idx_t) * nnodes);
            for (idx_t i = 0; i < nnodes; i++)
            {
                vdegrees[i] = m_origin_graph.row_start[i+1] - m_origin_graph.row_start[i];
            }

            printf("Degree computed, calling METIS\n"); 
            
            int result = METIS_PartGraphKway(
                &nnodes,                      // 
                &ncons,                       //
                row_start.data(),     //
                edge_dst.data(),      //
                vdegrees,                         // vwgt
                NULL,                         // vsize
                m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  // adjwgt
                &nparts,                      // nparts
                NULL,                         // tpwgts
                NULL,                         // ubvec
                NULL,                         // options
                &edgecut,                     // objval
                &partition_table[0]);         // part [out]

            if (result != METIS_OK) {
                printf(
                    "METIS partitioning failed (%s error), Exiting.\n", 
                    result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
                exit(0);
            }
            // free(vdegrees);

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("METIS partitioning done\n");
#endif
        }

        void MetisPartitionerDegreeW::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> MetisPartitionerDegreeW::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }

/* --------------------------------- NVGraph -------------------------------- */
        void check_status(nvgraphStatus_t status)
        {
            // printf("check status\n");
            if (status != NVGRAPH_STATUS_SUCCESS)
            {
                printf("ERROR : %s\n", nvgraphStatusGetString(status));
                exit(0);
            }
        }

        NVGraphPartitioner::NVGraphPartitioner(host::CSRGraph& origin_graph, int nsegs) : 
            m_origin_graph(origin_graph), 
            m_partitioned_graph(origin_graph.nnodes, origin_graph.nedges), 
            m_reverse_lookup(origin_graph.nnodes), m_seg_offsets(nsegs + 1),
            m_nsegs(nsegs)
        {
            printf("\nStarting NVGraph partitioning with vertex weights (degree)\n");

            int nnodes = m_origin_graph.nnodes;
            int nedges = m_origin_graph.nedges;

            int ncons = 1;
            int nparts = m_nsegs;

            printf("0..");
            fflush(stdout);
            int edgecut;
            std::vector<int> partition_table(nnodes);

            printf("1..");
            fflush(stdout);

            // Convert to int for nvgraph
            std::vector<int> row_start (nnodes+1), edge_dst (nedges), edge_weights;
            for (uint32_t i = 0; i < nnodes + 1; ++i)
                row_start[i] = static_cast<int>(m_origin_graph.row_start[i]);
            printf("2..");fflush(stdout);
            for (uint32_t i = 0; i < nedges; ++i)
                edge_dst[i] = static_cast<int>(m_origin_graph.edge_dst[i]);
            printf("3..");fflush(stdout);
            if(m_origin_graph.edge_weights)
            {
                edge_weights.resize(nedges);
                for (uint32_t i = 0; i < nedges; ++i)
                    edge_weights[i] = static_cast<int>(m_origin_graph.edge_weights[i]);
            }
            printf("Converted graph to %d-bit, calling nvgraph...\n", (int)IDXTYPEWIDTH);

            // idx_t *vdegrees = (idx_t*) malloc(sizeof(idx_t) * nnodes);
            // for (idx_t i = 0; i < nnodes; i++)
            // {
            //     vdegrees[i] = m_origin_graph.row_start[i+1] - m_origin_graph.row_start[i];
            // }

            // printf("Degree computed, calling METIS\n"); 

            nvgraphHandle_t handle;
            nvgraphGraphDescr_t graph;
            nvgraphCSRTopology32I_st CSRType;
            CSRType.nvertices = nnodes;
            CSRType.nedges = ncons;
            CSRType.source_offsets = row_start.data();
            CSRType.destination_indices = edge_dst.data();

            SpectralClusteringParameter param;
            param.n_clusters = nparts;
            param.n_eig_vects = nparts;
            param.algorithm = NVGRAPH_BALANCED_CUT_LANCZOS;
            param.evs_tolerance = 0.0f; // default
            param.evs_max_iter = 0; // default
            param.kmean_tolerance = 0.0f;   // default
            param.kmean_max_iter = 0;   // default
            
            cudaDataType_t edge_t = CUDA_R_32F;

            // allocate
            float *edgewgt, *eigvals, *eigvec;
            edgewgt = (float*) malloc(sizeof(float) * ncons);
            eigvals = (float*) malloc(sizeof(float)*nparts);
            eigvec = (float*) malloc(sizeof(float)*nparts*nnodes);
            for (uint32_t i = 0; i < ncons; i++)
            {
                edgewgt[i] = 1;
            }
            printf("prepared...");fflush(stdout);

            check_status(nvgraphCreate(&handle));
            check_status(nvgraphCreateGraphDescr(handle, &graph));
            check_status(nvgraphSetGraphStructure(handle, graph, (void*)&CSRType, NVGRAPH_CSR_32));
            check_status(nvgraphAllocateEdgeData(handle, graph, 1, &edge_t));
            check_status(nvgraphSetEdgeData(handle, graph, (void*)edgewgt, 0));
            printf("running...");fflush(stdout);
            check_status(nvgraphSpectralClustering(handle, graph, 0, &param, &partition_table[0], eigvals, eigvec));

            check_status(nvgraphDestroyGraphDescr(handle, graph));
            check_status(nvgraphDestroy(handle));
            free(edgewgt);
            free(eigvals);
            free(eigvec);
            
            // int result = METIS_PartGraphKway(
            //     &nnodes,                      // 
            //     &ncons,                       //
            //     row_start.data(),     //
            //     edge_dst.data(),      //
            //     NULL,                         // vwgt
            //     NULL,                         // vsize
            //     m_origin_graph.edge_weights ? edge_weights.data() : nullptr,  // adjwgt
            //     &nparts,                      // nparts
            //     NULL,                         // tpwgts
            //     NULL,                         // ubvec
            //     NULL,                         // options
            //     &edgecut,                     // objval
            //     &partition_table[0]);         // part [out]

            // if (result != METIS_OK) {
            //     printf(
            //         "METIS partitioning failed (%s error), Exiting.\n", 
            //         result == METIS_ERROR_INPUT ? "input" : result == METIS_ERROR_MEMORY ? "memory" : "general");
            //     exit(0);
            // }
            // free(vdegrees);

            printf("Building partitioned graph and lookup tables\n");

            struct node_partition {
                    index_t node;
                    index_t partition;

                    node_partition(index_t node, index_t partition) : node(node), partition(partition) {}
                    node_partition() : node(-1), partition(-1) {}

                    inline bool operator< (const node_partition& rhs) const {
                        return partition < rhs.partition;
                    }
            };

            std::vector<node_partition> node_partitions(nnodes);

            for (index_t node = 0; node < nnodes; ++node)
            {
                node_partitions[node] = node_partition(node, partition_table[node]);
            }

            std::stable_sort(node_partitions.begin(), node_partitions.end());

            if (m_origin_graph.edge_weights != nullptr)
            {
                m_partitioned_graph.AllocWeights();
            }

            int current_seg = -1;

            for (index_t new_nidx = 0, edge_pos = 0; new_nidx < nnodes; ++new_nidx)
            {
                int seg = node_partitions[new_nidx].partition;
                while (seg > current_seg) // if this is true we have crossed the border to the next seg (looping with while just in case)
                {
                    m_seg_offsets[++current_seg] = new_nidx;
                }

                index_t origin_nidx = node_partitions[new_nidx].node; 
                m_reverse_lookup[origin_nidx] = new_nidx;

                index_t edge_start = m_origin_graph.row_start[origin_nidx];
                index_t edge_end = m_origin_graph.row_start[origin_nidx+1];

                m_partitioned_graph.row_start[new_nidx] = edge_pos;

                std::copy(m_origin_graph.edge_dst + edge_start, m_origin_graph.edge_dst + edge_end, m_partitioned_graph.edge_dst + edge_pos);

                if (m_origin_graph.edge_weights != nullptr) // copy weights
                    std::copy(m_origin_graph.edge_weights + edge_start, m_origin_graph.edge_weights + edge_end, m_partitioned_graph.edge_weights + edge_pos);

                edge_pos += (edge_end - edge_start);
            }
            
            while (m_nsegs > current_seg) m_seg_offsets[++current_seg] = nnodes;

            m_partitioned_graph.row_start[nnodes] = nedges;
            
            // Map the original destinations, copied from the origin graph to the new index space
            for (index_t edge = 0; edge < nedges; ++edge)
            {
                index_t origin_dest = m_partitioned_graph.edge_dst[edge];
                m_partitioned_graph.edge_dst[edge] = m_reverse_lookup[origin_dest];
            }

            printf("NVGraph partitioning done\n");
        }

        void NVGraphPartitioner::GetSegIndices(
            int seg_idx,
            index_t& seg_snode, index_t& seg_nnodes,
            index_t& seg_sedge, index_t& seg_nedges) const
        {
            index_t seg_enode, seg_eedge;

            seg_snode = m_seg_offsets[seg_idx];
            seg_enode = m_seg_offsets[seg_idx + 1];
            seg_nnodes = seg_enode - seg_snode;                

            seg_sedge = m_partitioned_graph.row_start[seg_snode];                            // start edge
            seg_eedge = m_partitioned_graph.row_start[seg_enode];                            // end edge
            seg_nedges = seg_eedge - seg_sedge;  
        }
        
        std::function<index_t(index_t)> NVGraphPartitioner::GetReverseLookupFunc()
        {
            return [this](index_t idx) { return this->m_reverse_lookup[idx]; };
        }
    }   // namespace multi
}
}
